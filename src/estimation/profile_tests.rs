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
    // A `correlation` line's blocks, as `fit()` assigns them (#2110); the other arms keep
    // `prepare_run`'s population.
    let p = if p.parsed.model.residual_serial.is_some() {
        scored(p)
    } else {
        p
    };
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
    // The predicate above says the model *may* take the analytic gradient; the decline log
    // says whether every subject did. A profiled fit never turns this log into a warning
    // (`optimize_population` does, `optimize_profiled` does not), so this is the one place
    // a subject that fell back to FD shows.
    if analytic {
        assert_eq!(
            declines.declined_indices(),
            Vec::<usize>::new(),
            "subjects that declined the analytic outer gradient at runtime"
        );
    }
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
        eprintln!(
            "DBG {i}: handed {:.9e} fd {:.9e} held {:.9e}",
            g[i], fd, held
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
/// realised 1.78e-4 (analytic) and 5.5e-5 (FD); ~3× headroom. The `car1` arm (#2110): 1.3e-5,
/// identity 2.9e-5 (macOS arm64); the solve on a diagonal `R` against the dense production
/// gradient puts it at 0.87.
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

/// F2 with its arm kappa and `correlation car1(RHO) by OCC` (#2110): the envelope
/// gradient — the analytic adjoint over the IOV dense-`R` marginal (#2101), every subject
/// on it — is the gradient of the profile objective with the line, ρ's coordinate
/// included. Dies when the production gradient and the profile solve read different
/// covariances: the handed gradient then differentiates one objective and the FD another.
#[test]
fn the_envelope_gradient_is_the_profile_gradient_with_a_correlation_line() {
    envelope_matches_fd(
        |t| {
            assert!(t.contains("  omega ETA_EM") && t.contains("weight = SE\n"));
            t.replace(
                "  omega ETA_EM",
                "  theta RHO(0.6, 0.0, 0.95)\n  omega ETA_EM",
            )
            .replace(
                "weight = SE\n",
                "weight = SE\n  correlation car1(RHO) by OCC\n",
            )
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

/// #2285: every leg of a profiled gradient fit runs at `xtol_rel ≤`
/// [`PROFILED_GRADIENT_XTOL_REL`], whatever looser `outer_xtol` is set (the default
/// 1e-4 let one short L-BFGS step end #2280's A2 fit 1.584e-3 above the optimum); a
/// tighter user value wins; BOBYQA keeps `outer_xtol`. Dies when the ceiling is deleted
/// or applied to BOBYQA.
#[test]
fn a_profiled_gradient_fit_caps_its_xtol() {
    let default = FitOptions::default();
    assert_eq!(default.outer_xtol, 1e-4, "the default this caps moved");
    let looser = FitOptions {
        outer_xtol: 1e-2,
        ..FitOptions::default()
    };
    let tighter = FitOptions {
        outer_xtol: 1e-9,
        ..FitOptions::default()
    };
    for algo in [nlopt::Algorithm::Lbfgs, nlopt::Algorithm::Slsqp] {
        assert_eq!(profiled_xtol(algo, &default), PROFILED_GRADIENT_XTOL_REL);
        assert_eq!(profiled_xtol(algo, &looser), PROFILED_GRADIENT_XTOL_REL);
        assert_eq!(profiled_xtol(algo, &tighter), 1e-9);
    }
    assert!(PROFILED_GRADIENT_XTOL_REL < default.outer_xtol);
    let bobyqa = nlopt::Algorithm::Bobyqa;
    assert_eq!(profiled_xtol(bobyqa, &default), 1e-4);
    assert_eq!(profiled_xtol(bobyqa, &looser), 1e-2);
}

fn ok(s: nlopt::SuccessState) -> NloptReturn {
    Ok((s, 101.9))
}

fn err(f: nlopt::FailState) -> NloptReturn {
    Err((f, 101.9))
}

/// Which profiled stops get a verification leg: every stop that could be a minimum,
/// `Failure` and `RoundoffLimited` included (`main`'s L-BFGS returns `Failure` at the
/// optimum, #2285 M5); not a spent budget, a cancelled run, a bad argument, or BOBYQA.
/// A `Failure` / `RoundoffLimited` that has not left the start is not verified (#2338
/// review #3: a first step that fails would otherwise be verified by a leg that gains
/// nothing); a tolerance stop is verified either way. Dies when the `left_init` gate is
/// dropped, or applied to tolerance stops too.
#[test]
fn a_profiled_stop_that_could_be_a_minimum_is_verified() {
    use nlopt::{FailState as F, SuccessState as S};
    for algo in [nlopt::Algorithm::Lbfgs, nlopt::Algorithm::Slsqp] {
        for r in [
            ok(S::Success),
            ok(S::FtolReached),
            ok(S::XtolReached),
            ok(S::StopValReached),
            err(F::Failure),
            err(F::RoundoffLimited),
        ] {
            assert!(
                profiled_stop_needs_verify(algo, &r, false, true),
                "{algo:?} {r:?}"
            );
            assert!(
                !profiled_stop_needs_verify(algo, &r, true, true),
                "cancelled {r:?}"
            );
        }
        for r in [
            ok(S::Success),
            ok(S::FtolReached),
            ok(S::XtolReached),
            ok(S::StopValReached),
        ] {
            assert!(
                profiled_stop_needs_verify(algo, &r, false, false),
                "a tolerance stop at the start: {algo:?} {r:?}"
            );
        }
        for r in [err(F::Failure), err(F::RoundoffLimited)] {
            assert!(
                !profiled_stop_needs_verify(algo, &r, false, false),
                "a failure at the start: {algo:?} {r:?}"
            );
        }
        for r in [
            ok(S::MaxEvalReached),
            ok(S::MaxTimeReached),
            err(F::ForcedStop),
            err(F::InvalidArgs),
            err(F::OutOfMemory),
        ] {
            assert!(
                !profiled_stop_needs_verify(algo, &r, false, true),
                "{algo:?} {r:?}"
            );
        }
    }
    let bobyqa = nlopt::Algorithm::Bobyqa;
    assert!(!profiled_stop_needs_verify(
        bobyqa,
        &ok(S::FtolReached),
        false,
        true
    ));
    assert!(!profiled_stop_needs_verify(
        bobyqa,
        &err(F::Failure),
        false,
        true
    ));
}

/// The verdict comes from the verification gain, not from NLopt's status: a settled
/// leg is converged on `Failure` as on `XtolReached`; a leg cap, a spent budget or a leg
/// that could not start is not; with nothing verified the status decides (BOBYQA keeps
/// today's rule, #2336). Dies when `Failure` is mapped to unconverged again.
#[test]
fn a_profiled_verdict_is_the_verification_gain() {
    use nlopt::{FailState as F, SuccessState as S};
    let lbfgs = nlopt::Algorithm::Lbfgs;
    let verdict = |end, r: NloptReturn| profiled_converged(lbfgs, end, &r, false, true);
    // A `Failure` that never left the start is judged on its status, not converged.
    assert!(!profiled_converged(
        lbfgs,
        PolishEnd::NotNeeded,
        &err(F::Failure),
        false,
        false
    ));
    assert!(verdict(PolishEnd::Settled, err(F::Failure)));
    assert!(verdict(PolishEnd::Settled, err(F::RoundoffLimited)));
    assert!(verdict(PolishEnd::Settled, ok(S::XtolReached)));
    assert!(verdict(PolishEnd::Settled, ok(S::FtolReached)));
    // A leg that ended on its budget settles the loop, but verified nothing.
    assert!(!verdict(PolishEnd::Settled, ok(S::MaxEvalReached)));
    assert!(!verdict(PolishEnd::NotNeeded, ok(S::MaxEvalReached)));
    assert!(!verdict(PolishEnd::NotNeeded, err(F::ForcedStop)));
    for r in [ok(S::XtolReached), err(F::Failure)] {
        assert!(!verdict(
            PolishEnd::LegCapReached { last_gain: 2e-5 },
            r.clone()
        ));
        assert!(!verdict(PolishEnd::BudgetSpent, r.clone()));
        assert!(!verdict(PolishEnd::Skipped, r));
    }
    let bobyqa = nlopt::Algorithm::Bobyqa;
    let r = ok(S::FtolReached);
    assert!(profiled_converged(
        bobyqa,
        PolishEnd::NotNeeded,
        &r,
        false,
        true
    ));
    assert!(!profiled_converged(
        bobyqa,
        PolishEnd::NotNeeded,
        &err(F::Failure),
        false,
        true
    ));
}

/// Each warning a profiled verdict emits, sentence by sentence, over every way the
/// verification can end (#2338 review #1, #6): the leg cap names the legs, the last gain
/// and that the fit is not converged; a budget spent before a leg and one spent during a
/// leg (`Settled` on `MaxEvalReached`) both get the budget advice; any other unconverged
/// stop names its status; a converged fit gets nothing.
#[test]
fn a_profiled_verdict_names_why_it_is_not_converged() {
    use nlopt::{FailState as F, SuccessState as S};
    let w = |end, r: NloptReturn, converged| profiled_verdict_warning(end, &r, converged);

    let cap = w(
        PolishEnd::LegCapReached { last_gain: 2.1e-5 },
        ok(S::XtolReached),
        false,
    )
    .expect("a leg cap warns");
    for sentence in [
        format!("still improving after {MAX_POLISH_LEGS} verification legs"),
        "the last lowered the objective by 2.100e-5".to_string(),
        "so it is not reported as converged".to_string(),
        "re-run from these estimates".to_string(),
    ] {
        assert!(cap.contains(&sentence), "{sentence:?} missing from {cap:?}");
    }

    let budget = w(PolishEnd::BudgetSpent, ok(S::FtolReached), false).expect("a spent budget");
    for sentence in [
        "spent its evaluation budget",
        "before its stop was verified",
        "increase maxiter",
    ] {
        assert!(
            budget.contains(sentence),
            "{sentence:?} missing from {budget:?}"
        );
    }
    assert_eq!(
        w(PolishEnd::Settled, ok(S::MaxEvalReached), false).as_deref(),
        Some(budget.as_str()),
        "a budget spent during a leg"
    );

    assert_eq!(
        w(PolishEnd::NotNeeded, err(F::Failure), false).as_deref(),
        Some("Profiled outer optimizer stopped without converging (Failure).")
    );
    assert_eq!(
        w(PolishEnd::NotNeeded, ok(S::MaxEvalReached), false).as_deref(),
        Some("Profiled outer optimizer stopped without converging (MaxEvalReached).")
    );
    assert_eq!(
        w(PolishEnd::Skipped, ok(S::XtolReached), false).as_deref(),
        Some("Profiled outer optimizer stopped without converging (XtolReached).")
    );

    assert_eq!(w(PolishEnd::Settled, err(F::Failure), true), None);
    assert_eq!(w(PolishEnd::NotNeeded, ok(S::FtolReached), true), None);
}

/// The loop the verdict reads, driven by the profiled predicate over a scripted run:
/// a `Failure` stop is verified by a leg, a leg that still gains more than
/// [`PROFILED_VERIFY_GAIN`] is followed by another, and the fit is converged on the leg
/// that gains less — on whatever status that leg ended. A run whose every leg gains more
/// ends on the leg cap, not converged. Dies when `Failure` is not verified (the stop is
/// then judged on its status) or when a large gain settles the loop.
#[test]
fn a_profiled_stop_is_converged_when_a_leg_gains_nothing() {
    use nlopt::{FailState as F, SuccessState as S};
    let lbfgs = nlopt::Algorithm::Lbfgs;
    let mut script = vec![(ok(S::XtolReached), 2.1e-5), (err(F::Failure), 6e-8)].into_iter();
    let (last, end) = drive_polish_legs(
        err(F::Failure),
        PROFILED_VERIFY_GAIN,
        |r| profiled_stop_needs_verify(lbfgs, r, false, true),
        || 100,
        |_| script.next(),
    );
    assert_eq!(end, PolishEnd::Settled);
    assert!(script.next().is_none(), "both legs ran");
    assert!(profiled_converged(lbfgs, end, &last, false, true));

    let (last, end) = drive_polish_legs(
        ok(S::FtolReached),
        PROFILED_VERIFY_GAIN,
        |r| profiled_stop_needs_verify(lbfgs, r, false, true),
        || 100,
        |_| Some((ok(S::XtolReached), 2.1e-5)),
    );
    assert_eq!(end, PolishEnd::LegCapReached { last_gain: 2.1e-5 });
    assert!(!profiled_converged(lbfgs, end, &last, false, true));

    // A `Failure` that never left the start runs no leg and is not converged.
    let (last, end) = drive_polish_legs(
        err(F::Failure),
        PROFILED_VERIFY_GAIN,
        |r| profiled_stop_needs_verify(lbfgs, r, false, false),
        || 100,
        |_| -> Option<(NloptReturn, f64)> { panic!("no leg may run from a failed start") },
    );
    assert_eq!(end, PolishEnd::NotNeeded);
    assert!(!profiled_converged(lbfgs, end, &last, false, false));
}

/// The verification legs [`optimize_profiled`] ran on this thread since the last call.
fn take_verify_legs() -> Vec<VerifyLeg> {
    VERIFY_LEGS.with(|v| std::mem::take(&mut *v.borrow_mut()))
}

/// The legs of one profiled fit: each runs at the capped `xtol_rel`, read back from the
/// leg itself, and reports the gain the incumbent actually made.
fn assert_legs_are_capped_and_honest(legs: &[VerifyLeg], o: &FitOptions) {
    let want = profiled_xtol(nlopt::Algorithm::Lbfgs, o);
    assert!(want <= PROFILED_GRADIENT_XTOL_REL);
    for (k, l) in legs.iter().enumerate() {
        assert_eq!(l.xtol, Some(want), "leg {k} ran at another xtol_rel: {l:?}");
        assert!(
            l.before.is_finite() && l.after.is_finite(),
            "leg {k}: {l:?}"
        );
        assert_eq!(
            l.gain,
            l.before - l.after,
            "leg {k} reported a gain it did not make"
        );
    }
}

/// A profiled L-BFGS fit runs its verification leg and ends converged: F2 on its first
/// two studies, through [`optimize_profiled`] itself, every leg at the capped `xtol_rel`
/// and reporting its real gain. Dies when the leg is not run, or runs at `outer_xtol`
/// (#2338 review #2, M1).
#[test]
fn a_profiled_fit_runs_its_verification_leg() {
    let p = prep(F2, F2_DATA, Some(2), |t| t);
    let o = FitOptions {
        run_covariance_step: false,
        ..options(&p)
    };
    assert!(matches!(profiled_algorithm(&o), nlopt::Algorithm::Lbfgs));
    take_verify_legs();
    let r = optimize_profiled(&p.parsed.model, &p.population, &p.init_params, &o);
    let legs = take_verify_legs();
    eprintln!("F2 legs: {legs:?}");
    assert!(!legs.is_empty(), "no verification leg ran");
    assert_legs_are_capped_and_honest(&legs, &o);
    let last = legs.last().unwrap();
    assert!(last.gain <= PROFILED_VERIFY_GAIN, "{legs:?}");
    assert!(r.converged, "{} legs: {:?}", legs.len(), r.warnings);
}

/// A stop far from the optimum is followed by legs while they still gain more than
/// [`PROFILED_VERIFY_GAIN`]: F2 with a user `outer_ftol` loose enough that every run,
/// legs included, stops well short of the optimum. Each leg gains more than the
/// threshold, so the loop reaches the leg cap and the fit is not converged, with the
/// leg-cap warning. Dies when a leg's gain is not what decides another leg (#2338
/// review #2, M2: the loop settles after one leg and the fit is called converged).
///
/// Measured (macOS arm64): the four legs gain 3.48e-5, 4.58e-2, 4.86e-3 and 3.32e-4 from
/// OFV 22.234048, the smallest 35× the threshold; the fit ends at 22.183013, 0.043 above
/// the 22.139638 the default run reaches with one leg gaining 2.3e-10.
#[test]
fn a_profiled_fit_whose_legs_keep_gaining_is_not_converged() {
    let p = prep(F2, F2_DATA, Some(2), |t| t);
    let o = FitOptions {
        run_covariance_step: false,
        outer_ftol: Some(LOOSE_FTOL),
        ..options(&p)
    };
    take_verify_legs();
    let r = optimize_profiled(&p.parsed.model, &p.population, &p.init_params, &o);
    let legs = take_verify_legs();
    eprintln!("F2 loose-ftol legs: {legs:?}; warnings {:?}", r.warnings);
    assert_legs_are_capped_and_honest(&legs, &o);
    assert_eq!(legs.len(), MAX_POLISH_LEGS, "{legs:?}");
    assert!(
        legs.iter().all(|l| l.gain > PROFILED_VERIFY_GAIN),
        "{legs:?}"
    );
    assert!(!r.converged);
    assert_eq!(
        r.warnings
            .iter()
            .filter(|w| w.contains("verification legs"))
            .count(),
        1,
        "{:?}",
        r.warnings
    );
}

const LOOSE_FTOL: f64 = 1e-3;

// ── #2136: the unprofiled loop's machinery, in the profiled loop ───────────────────

/// F2 on two studies, run through [`optimize_profiled`] for a few iterations with no
/// covariance step: what the #2136 rows below inspect.
fn short_profiled_fit(edit: impl Fn(String) -> String) -> (crate::api::PreparedRun, OuterResult) {
    let p = prep(F2, F2_DATA, Some(2), edit);
    let o = FitOptions {
        outer_maxiter: 3,
        run_covariance_step: false,
        ..options(&p)
    };
    let r = optimize_profiled(&p.parsed.model, &p.population, &p.init_params, &o);
    (p, r)
}

/// F2 with a structural θ nothing reads: `JUNK` feeds an individual parameter the model
/// never uses, the #826 shape.
fn with_junk_theta(t: String) -> String {
    t.replace(
        "  theta ET50(2.0, 0.01, 100.0)\n",
        "  theta ET50(2.0, 0.01, 100.0)\n  theta JUNK(1.0, 0.01, 10.0)\n",
    )
    .replace(
        "  EMAXI = EMAX * exp(ETA_EM)\n",
        "  EMAXI = EMAX * exp(ETA_EM)\n  UNUSED = JUNK\n",
    )
}

/// Row 1, the flat-θ pre-flight (#826): a structural θ the profile objective ignores is
/// frozen at its start with the unprofiled loop's warning, and published `FIX`; every θ
/// the objective reads stays free. Dies when the pre-flight is not run, or freezes on the
/// gradient alone without its probe confirming (the identifiable θ would then have to
/// sit at a zero gradient, which none does here — so it is the `JUNK` assertions that
/// carry the row).
#[test]
fn a_profiled_fit_freezes_a_theta_the_objective_ignores() {
    let (p, r) = short_profiled_fit(with_junk_theta);
    let names = &p.init_params.theta_names;
    let junk = names
        .iter()
        .position(|n| n == "JUNK")
        .expect("JUNK declared");
    assert!(
        !p.init_params.theta_fixed[junk],
        "fixture: JUNK starts free"
    );
    let w = crate::estimation::outer_optimizer::flat_theta_warning("JUNK", 1.0);
    assert!(r.warnings.contains(&w), "{:?}", r.warnings);
    assert!(r.params.theta_fixed[junk]);
    assert_eq!(r.params.theta[junk], 1.0);
    for name in ["TVE0", "EMAX", "ET50"] {
        let k = names.iter().position(|n| n == name).unwrap();
        assert!(!r.params.theta_fixed[k], "{name} must stay free");
        assert!(
            !r.warnings
                .iter()
                .any(|w| w.contains(&format!("`{name}` has no effect"))),
            "{name}: {:?}",
            r.warnings
        );
    }
    // The levels stay estimated parameters in the published template (#2103 review #9).
    let layout = layout_of(&p);
    assert!(layout.coords.iter().all(|&k| !r.params.theta_fixed[k]));
}

/// Row 2, checkpoints (#755): with a sink due on every write, a profiled fit leaves a
/// checkpoint of the fit's own packed layout whose levels are a solved `β̂` (the template
/// holds them at 0), and re-profiling from it reproduces its recorded objective — what a
/// resume starts from. Dies when the profiled loop writes no checkpoint, or writes the
/// optimizer's point with the levels still at the template.
///
/// And a due sink is written after **every** evaluation, improving or not, with the
/// incumbent: the fixture is asserted to have evaluations that did not replace the
/// incumbent, so a write keyed on an improvement (round 1 of #2136's review: with a
/// positive interval that never writes once the incumbent predates the interval) writes
/// fewer times than there were evaluations and fails. The last write holds the final
/// incumbent, so its objective is the fit's own, bit for bit.
#[test]
fn a_profiled_fit_writes_a_checkpoint_at_its_incumbent() {
    let path = temp("checkpoint.json");
    let path_s = path.to_str().unwrap().to_string();
    let p = prep(F2, F2_DATA, Some(2), |t| t);
    crate::io::checkpoint::init(
        path_s.clone(),
        "test".into(),
        None,
        None,
        vec!["focei".into()],
        crate::estimation::parameterization::coordinate_names(&p.init_params),
        0,
    );
    let o = FitOptions {
        outer_maxiter: 10,
        run_covariance_step: false,
        ..options(&p)
    };
    let writes_before = crate::io::checkpoint::writes();
    let updates_before = INCUMBENT_UPDATES.with(std::cell::Cell::get);
    let r = optimize_profiled(&p.parsed.model, &p.population, &p.init_params, &o);
    let writes = crate::io::checkpoint::writes() - writes_before;
    let updates = INCUMBENT_UPDATES.with(std::cell::Cell::get) - updates_before;
    crate::io::checkpoint::abandon();
    let cp = crate::io::checkpoint::load(&path_s).expect("a checkpoint was written");
    crate::io::checkpoint::remove(&path_s);
    // Measured (macOS arm64): 45 evaluations, 39 incumbent updates, 45 writes.
    eprintln!(
        "#2136 checkpoint: {} evaluations, {updates} incumbent updates, {writes} writes",
        r.n_iterations
    );
    assert!(
        updates >= 1 && updates < r.n_iterations,
        "fixture: some evaluation must not improve, or the straddle is empty"
    );
    // The first evaluation is feasible here, so an incumbent exists after every one.
    assert_eq!(writes, r.n_iterations, "a write after every evaluation");
    assert_eq!(cp.ofv.to_bits(), r.ofv.to_bits(), "the final incumbent");
    assert!(cp.iter >= 1 && cp.iter <= r.n_iterations, "{}", cp.iter);

    assert_eq!(
        cp.packed.len(),
        crate::estimation::parameterization::packed_len(&p.init_params)
    );
    let layout = layout_of(&p);
    let at = unpack_params(&cp.packed, &p.init_params);
    assert!(
        layout.coords.iter().any(|&k| at.theta[k] != 0.0),
        "levels at the template: {:?}",
        layout
            .coords
            .iter()
            .map(|&k| at.theta[k])
            .collect::<Vec<_>>()
    );
    assert!(
        cp.ofv.is_finite() && cp.ofv >= r.ofv - 1e-9,
        "{} vs {}",
        cp.ofv,
        r.ofv
    );
    let c = ctx(&p, &layout, &o);
    let warm = ProfileWarm {
        beta: layout.coords.iter().map(|&k| at.theta[k]).collect(),
        etas: None,
    };
    let s = c.solve(&at, &warm).expect("the checkpoint re-profiles");
    assert!(
        (s.ofv - cp.ofv).abs() < 1e-6 * (1.0 + cp.ofv.abs()),
        "{} vs {}",
        s.ofv,
        cp.ofv
    );
}

/// Row 3, the optimizer trace: a profiled fit writes one row per evaluation, each with
/// every packed coordinate's value — the levels at a solved `β̂`, not the template's 0 —
/// and the gradient NLopt was handed on the structural coordinates, `NA` on the held
/// levels. Dies when the profiled loop writes no rows, or writes the held levels'
/// (unread) gradient entries.
#[test]
fn a_profiled_fit_writes_its_optimizer_trace() {
    let path = temp("trace.csv");
    let path_s = path.to_str().unwrap().to_string();
    let p = prep(F2, F2_DATA, Some(2), |t| t);
    let names = crate::estimation::parameterization::coordinate_names(&p.init_params);
    crate::estimation::trace::init(path_s.clone(), &names).unwrap();
    let o = FitOptions {
        outer_maxiter: 3,
        run_covariance_step: false,
        ..options(&p)
    };
    let r = optimize_profiled(&p.parsed.model, &p.population, &p.init_params, &o);
    crate::estimation::trace::finish();
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::remove_file(&path).ok();
    let rows: Vec<Vec<&str>> = text
        .lines()
        .skip(1)
        .map(|l| l.split(',').collect())
        .collect();
    assert!(
        !rows.is_empty() && rows.len() <= r.n_iterations,
        "{} rows",
        rows.len()
    );
    let n = names.len();
    let layout = layout_of(&p);
    let tve0 = names.iter().position(|n| n == "TVE0").unwrap();
    for row in &rows {
        assert_eq!(row.len(), 17 + 2 * n, "{row:?}");
        assert!(row[3].parse::<f64>().unwrap().is_finite(), "ofv {row:?}");
        assert_ne!(row[17 + tve0 + n], "NA", "a structural gradient: {row:?}");
        for &k in &layout.coords {
            assert_eq!(row[17 + k + n], "NA", "a held level's gradient: {row:?}");
        }
        assert!(
            layout
                .coords
                .iter()
                .any(|&k| row[17 + k].parse::<f64>().unwrap() != 0.0),
            "the levels are solved: {row:?}"
        );
    }
}

/// Row 4, the non-PD SIR fallback: an `A_p` that is not positive definite publishes no
/// matrix, warns with its eigenvalues, and hands SIR the unprofiled step's rectified
/// proposal over the structural block — at the same 4× inflation — extended over the
/// levels by the same block inverse with `D⁻¹` inflated alike, so the proposal is a
/// covariance over every packed coordinate. Both sides of the gate in one test: the
/// positive-definite `A_p` the stencil measured publishes `2·A_p⁻¹` and no proposal. The
/// indefinite one is that matrix with one eigenvalue negated; `D` and `B` are real, from
/// the fixture's level blocks. Dies when the proposal is dropped (`None`, the state
/// before #2136), or built without the level blocks.
#[test]
fn a_non_pd_profile_hessian_hands_sir_a_fallback_proposal() {
    let p = prep(F2, F2_DATA, Some(2), |t| t);
    let layout = layout_of(&p);
    let o = options(&p);
    let c = ctx(&p, &layout, &o);
    let hold = hold_template(&layout, &p.init_params);
    let PackedStart {
        packed: x, fixed, ..
    } = pack_with_bounds(&hold);
    let free: Vec<usize> = (0..x.len()).filter(|&i| !fixed[i]).collect();
    let s = c
        .solve(
            &unpack_params(&x, &hold),
            &ProfileWarm::from_template(&layout, &p.init_params),
        )
        .unwrap();
    let x_hat = solved_point(&layout, &hold, &x, &s);
    let nf = free.len();
    // A PD A_p of the fixture's scale: the diagonal of a structural FD stencil, not
    // needed exactly — the claim is about the branch, and D, B are the real ones.
    let a_pd = DMatrix::from_fn(
        nf,
        nf,
        |a, b| if a == b { 10.0 * (a + 1) as f64 } else { 0.5 },
    );
    let pd = c.covariance_from_profile_hessian(&hold, &x_hat, &free, &s, &a_pd, Instant::now());
    let m = pd.matrix.expect("a PD A_p publishes a matrix");
    assert!(pd.sir_fallback_proposal.is_none() && pd.warnings.is_empty());
    let want_s = a_pd.clone().cholesky().unwrap().inverse() * 2.0;
    for a in 0..nf {
        for b in 0..nf {
            assert!((m[(free[a], free[b])] - want_s[(a, b)]).abs() < 1e-12);
        }
    }

    let mut eig = nalgebra::SymmetricEigen::new(a_pd.clone());
    eig.eigenvalues[0] = -eig.eigenvalues[0];
    let a_bad = eig.recompose();
    let out = c.covariance_from_profile_hessian(&hold, &x_hat, &free, &s, &a_bad, Instant::now());
    assert!(out.matrix.is_none() && out.method.is_none());
    let eigvals = crate::estimation::covariance::extract_eigenvalues(&a_bad).unwrap();
    assert_eq!(out.warnings, vec![profile_non_pd_msg(&eigvals)]);
    let prop = out
        .sir_fallback_proposal
        .expect("a non-PD A_p hands SIR a proposal");
    assert_eq!((prop.nrows(), prop.ncols()), (x.len(), x.len()));
    // The structural block is the unprofiled builder's.
    let units = crate::estimation::covariance::free_coordinate_units(
        &hold,
        &x_hat,
        &free,
        &p.parsed.model.nn_weight_theta_ranges(),
    );
    let all: Vec<usize> = (0..nf).collect();
    let want = crate::estimation::covariance::build_non_pd_fallback_proposal(
        &a_bad, &all, nf, 4.0, &units,
    );
    for a in 0..nf {
        for b in 0..nf {
            assert!(
                (prop[(free[a], free[b])] - want[(a, b)]).abs() < 1e-12,
                "({a}, {b})"
            );
        }
    }
    // The levels carry variance, and the proposal is a covariance SIR can draw from.
    assert!(layout.coords.iter().all(|&k| prop[(k, k)] > 0.0));
    let idx: Vec<usize> = free.iter().chain(&layout.coords).copied().collect();
    let sub = DMatrix::from_fn(idx.len(), idx.len(), |a, b| prop[(idx[a], idx[b])]);
    assert!((&sub - sub.transpose()).amax() < 1e-9 * sub.amax());
    assert!(
        sub.cholesky().is_some(),
        "the proposal is positive definite"
    );
    // Its level blocks are twice the PD covariance's `D⁻¹` term: rebuilt from the same
    // structural block, `d_scale = 4` against `d_scale = 2` differs by exactly `2·D⁻¹`.
    let (d_groups, b) = c.level_blocks(&hold, &x_hat, &free, &s).unwrap();
    let ps = DMatrix::from_fn(nf, nf, |a, b| prop[(free[a], free[b])]);
    let at2 =
        assemble_profiled_covariance(&layout, &free, x.len(), &ps, &d_groups, &b, 2.0).unwrap();
    let k0 = layout.coords[0];
    let g0 = layout.local[0].0;
    let d_inv = d_groups[g0].clone().cholesky().unwrap().inverse();
    let la = layout.local[0].1;
    assert!(
        ((prop[(k0, k0)] - at2[(k0, k0)]) - 2.0 * d_inv[(la, la)]).abs()
            < 1e-9 * (1.0 + prop[(k0, k0)].abs()),
        "{} {} {}",
        prop[(k0, k0)],
        at2[(k0, k0)],
        d_inv[(la, la)]
    );
}

/// #10, the envelope gradient's reconverge override, keyed on the route each subject
/// took. F2 without κ is analytic at model level; every subject is forced to decline the
/// analytic gradient at runtime, the numerical decline `sens/provider.rs` names. On the
/// reconverging log `optimize_profiled` builds — taken from the same
/// [`profiled_decline_log`] the driver calls, so a driver built on a plain log fails
/// here — the declined subjects are reconverged
/// and the gradient is still the profile gradient (G1's bound); on a plain log they take
/// the held-EBE salvage, which drops `dη̂/dθ`, and it is not — the straddle, asserted so
/// the test cannot pass on a fixture where the two agree. Dies when the profiled loop
/// keys the override on the model-level predicate alone (a plain log). Measured (F2,
/// three studies, macOS arm64): reconverged 2.1e-5, held-EBE 2.2 — the bound sits 24×
/// above one and 4500× below the other.
#[test]
fn a_subject_that_declines_at_runtime_gets_the_reconverged_envelope_gradient() {
    let p = prep(F2, F2_DATA, Some(3), |t| {
        t.replace("  kappa KAPPA_ARM ~ 1.0\n", "")
            .replace(" + KAPPA_ARM", "")
            .replace("  iov_column = OCC\n", "")
    });
    assert!(
        crate::sens::provider::analytic_outer_gradient_for_interaction(
            &p.parsed.model,
            p.parsed.fit_options.interaction
        ),
        "fixture: analytic at model level"
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
    let s = c
        .solve(
            &unpack_params(&x, &hold),
            &ProfileWarm::from_template(&layout, &p.init_params),
        )
        .unwrap();
    let w = ProfileWarm::from_solve(&layout, &s);
    let n_subj = p.population.subjects.len();
    let worst = |log: OuterFdDeclineLog| {
        for i in 0..n_subj {
            log.force_decline(i);
        }
        let g = envelope_gradient(&c, &hold, &x, &s, &bounds, &mut 0, &log);
        assert_eq!(
            log.declined_indices().len(),
            n_subj,
            "every subject declined"
        );
        let mut worst = 0.0_f64;
        for i in (0..x.len()).filter(|&i| !fixed[i]) {
            let h = 1e-4 * (1.0 + x[i].abs());
            let moved = |d: f64| {
                let mut y = x.clone();
                y[i] += d;
                y
            };
            let fd = (c.profile_ofv(&hold, &moved(h), &w) - c.profile_ofv(&hold, &moved(-h), &w))
                / (2.0 * h);
            assert!(fd.is_finite() && g[i].is_finite(), "{i}: {fd} / {}", g[i]);
            worst = worst.max((g[i] - fd).abs() / (1.0 + fd.abs()));
        }
        worst
    };
    let reconverged = worst(profiled_decline_log(n_subj));
    let held = worst(OuterFdDeclineLog::new(n_subj));
    eprintln!("#2136 #10: reconverged {reconverged:.3e}, held-EBE {held:.3e}");
    assert!(reconverged < G1_HANDED_BOUND, "{reconverged}");
    assert!(held > G1_HANDED_BOUND, "the straddle: held-EBE {held}");
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

// ── #2261: every re-solve of the profile starts at the previous (η̂, κ̂) ──────────────

/// The pass-`k` EBEs of a recording, per subject in population order.
fn pass_modes(
    rec: &crate::estimation::inner_optimizer::InnerRecording,
    k: usize,
    pop: &Population,
) -> (Vec<DVector<f64>>, Vec<Vec<DVector<f64>>>) {
    let pass: Vec<_> = rec.passes.iter().filter(|p| p.pass == k).collect();
    assert_eq!(pass.len(), pop.subjects.len(), "pass {k}");
    (
        pass.iter()
            .map(|p| DVector::from_column_slice(&p.eta))
            .collect(),
        pass.iter()
            .map(|p| {
                p.kappas
                    .iter()
                    .map(|c| DVector::from_column_slice(c))
                    .collect()
            })
            .collect(),
    )
}

/// F1 from the template takes one Newton step, so on its one solve every subject's
/// re-solves run in a fixed order: `2k` central-difference probes of `total_gradient`, then
/// the line search's first trial — all from the state the population pass returned — then
/// the second iteration's probes, from the trial's modes. The first `2k + 1` of them must
/// start at the pass's (η̂, κ̂): the gradient's warm start, then the line search's.
#[test]
fn profile_gradient_and_line_search_start_at_the_pass_mode() {
    use crate::estimation::inner_optimizer::record_inner_solves;
    use crate::estimation::inner_optimizer::test_helpers::{
        assert_these_start_at_modes, on_one_thread,
    };
    let p = prep(F1, F1_DATA, Some(2), |t| t);
    let layout = layout_of(&p);
    let o = options(&p);
    let c = ctx(&p, &layout, &o);
    let (s, rec) = on_one_thread(|| {
        record_inner_solves(|| {
            c.solve(
                &p.init_params,
                &ProfileWarm::from_template(&layout, &p.init_params),
            )
            .expect("the solve converges")
        })
    });
    assert_eq!(
        s.iterations, 1,
        "fixture: one Newton step, so one line search"
    );
    let (etas, kappas) = pass_modes(&rec, 0, &p.population);
    let mut first = Vec::new();
    let mut line_search = Vec::new();
    for (i, subject) in p.population.subjects.iter().enumerate() {
        let k = layout.reads[i].len();
        let mine: Vec<_> = rec
            .starts
            .iter()
            .filter(|s| s.pass.is_none() && s.subject == subject.id)
            .collect();
        if k == 0 {
            assert!(mine.is_empty(), "subject {} reads no level", subject.id);
            continue;
        }
        assert!(
            mine.len() > 2 * k,
            "subject {}: {} re-solves",
            subject.id,
            mine.len()
        );
        first.extend(mine[..2 * k].iter().copied());
        line_search.push(mine[2 * k]);
    }
    assert_these_start_at_modes(
        "profile total_gradient",
        first,
        &p.population,
        &etas,
        &kappas,
    );
    assert_these_start_at_modes(
        "profile line search",
        line_search,
        &p.population,
        &etas,
        &kappas,
    );
}

/// A solve warm-started from a previous one (`ProfileWarm::from_solve`, the outer loop's
/// next evaluation) starts its population pass at that solve's (η̂, κ̂), at moved
/// structural parameters.
#[test]
fn profile_solve_from_a_previous_solve_starts_at_its_mode() {
    use crate::estimation::inner_optimizer::record_inner_solves;
    use crate::estimation::inner_optimizer::test_helpers::{assert_starts_at_modes, on_one_thread};
    let p = prep(F1, F1_DATA, Some(2), |t| t);
    let layout = layout_of(&p);
    let o = options(&p);
    let c = ctx(&p, &layout, &o);
    let first = c
        .solve(
            &p.init_params,
            &ProfileWarm::from_template(&layout, &p.init_params),
        )
        .expect("the solve converges");
    let mut moved = p.init_params.clone();
    moved.theta[0] *= 1.01;
    let rec = on_one_thread(|| {
        record_inner_solves(|| {
            c.solve(&moved, &ProfileWarm::from_solve(&layout, &first))
                .expect("the solve converges")
        })
        .1
    });
    assert_starts_at_modes(
        "profile ProfileWarm::from_solve",
        &rec,
        |s| s.pass == Some(0),
        &p.population,
        &first.etas,
        &first.kappas,
    );
}

/// The profiled covariance step handed the fit's (η̂, κ̂) — what `fit()` and `run_covariance`
/// both hand it — starts its re-profiling pass there. Every later re-solve warm-starts from a
/// mode an earlier solve of the same subject returned, κ̂ included: the re-profiling solves
/// at each structural stencil point (`ProfileWarm::from_solve`), their gradient probes and
/// line searches, and the level stencil.
#[test]
fn profiled_covariance_starts_at_the_fit_mode() {
    use crate::estimation::inner_optimizer::record_inner_solves;
    use crate::estimation::inner_optimizer::test_helpers::{
        assert_starts_at_modes, assert_starts_at_previous_modes, on_one_thread,
    };
    let p = prep(F1, F1_DATA, Some(2), |t| t);
    let layout = layout_of(&p);
    let o = FitOptions {
        covariance_method: CovarianceMethod::Hessian,
        ..options(&p)
    };
    let c = ctx(&p, &layout, &o);
    let fit = c
        .solve(
            &p.init_params,
            &ProfileWarm::from_template(&layout, &p.init_params),
        )
        .expect("the solve converges");
    let x = pack_with_bounds(&fit.params).packed;
    let rec = on_one_thread(|| {
        record_inner_solves(|| {
            replay_covariance(
                &p.parsed.model,
                &p.population,
                &fit.params,
                &x,
                &fit.etas,
                &fit.kappas,
                &o,
            )
            .expect("F1 profiles a level block")
        })
        .1
    });
    assert!(
        rec.passes.iter().all(|p| p.init.is_some()),
        "every re-profiling solve is warm"
    );
    assert_starts_at_modes(
        "profiled covariance re-profile",
        &rec,
        |s| s.pass == Some(0),
        &p.population,
        &fit.etas,
        &fit.kappas,
    );
    let n =
        assert_starts_at_previous_modes("profiled covariance stencil", &rec, |s| s.pass != Some(0));
    assert!(
        n > rec.passes.len(),
        "the per-subject re-solves (level stencil, gradient probes) are among the {n}"
    );
}

// ── #2110: a residual `correlation` line with a profiled block ─────────────────────

/// F1 with `theta RHO(init, lo, 0.95)` and `correlation <kind>(RHO) by OCC` — the MBMA
/// shape that matters: the arm kappa `KAPPA_ARM` lives at `OCC`, and the line correlates
/// the visits of each arm (`car1` on the visit times, `ar1` on their order).
fn f1_corr(kind: &'static str, init: &'static str, lo: &'static str) -> impl Fn(String) -> String {
    move |t: String| {
        assert!(
            t.contains("  omega ETA_E0") && t.contains("weight = SE\n"),
            "F1 moved"
        );
        t.replace(
            "  omega ETA_E0",
            &format!("  theta RHO({init}, {lo}, 0.95)\n  omega ETA_E0"),
        )
        .replace(
            "weight = SE\n",
            &format!("weight = SE\n  correlation {kind}(RHO) by OCC\n"),
        )
    }
}

/// `nested_levels_3` (sponsor > study > arm, the arm and study kappas inside the
/// `exp`, so nonlinear in them) with a profiled `PLACEBO[STUDYID, TIME]` block on the
/// baseline and `correlation ar1(RHO) by ARM`, which groups by the nested arm level.
fn nested_corr(t: String) -> String {
    assert!(t.contains("  y = TVE0 + "), "nested_levels_3 moved");
    t.replace(
        "  omega ETA_SP",
        "  theta PLACEBO[STUDYID, TIME](0.0) profile\n  theta RHO(0.3, -0.95, 0.95)\n  \
         omega ETA_SP",
    )
    .replace("  y = TVE0 + ", "  y = TVE0 + PLACEBO + ")
    .replace(
        "DV ~ additive(ADD)",
        "DV ~ additive(ADD)\n  correlation ar1(RHO) by ARM",
    )
}

const NESTED: &str = "nonmem_anchor/nested_levels_3.ferx";
const NESTED_DATA: &str = "nonmem_anchor/nested_levels_3.csv";

/// `p` with the population `fit()` scores (`fitted_population`): a `correlation` line's
/// blocks and a nested model's level cells assigned. `prepare_run` assigns neither, and a
/// subject without its blocks scores a NaN `R`.
fn scored(mut p: crate::api::PreparedRun) -> crate::api::PreparedRun {
    p.population = crate::api::fitted_population(
        &p.parsed.model,
        &crate::types::IovOccasionRule::Column,
        true,
        std::borrow::Cow::Owned(p.population),
        &mut Vec::new(),
    )
    .into_owned();
    assert!(
        p.parsed.model.residual_serial.is_none()
            || p.population
                .subjects
                .iter()
                .all(|s| s.obs_rgroup.len() == s.obs_times.len())
    );
    p
}

/// The solve at the template, through [`ProfileCtx::solve`].
fn solve_at(p: &crate::api::PreparedRun) -> (ProfileLayout, ProfileSolve) {
    let layout = layout_of(p);
    let o = options(p);
    let s = ctx(p, &layout, &o)
        .solve(
            &p.init_params,
            &ProfileWarm::from_template(&layout, &p.init_params),
        )
        .unwrap_or_else(|e| panic!("the profile solve failed: {e:?}"));
    (layout, s)
}

/// N1 with the line: F1 is still linear in (η, κ) and the dense `R` does not read the
/// levels, so `XᵀC̃⁻¹X` is still the exact Hessian and the solve takes **one** step.
/// Dies when the Newton matrix is built without the line's correlation (`held_hessian`
/// on a diagonal `R`), which still converges, but in more steps.
#[test]
fn a_correlated_model_linear_in_its_random_effects_profiles_in_one_newton_step() {
    for kind in ["car1", "ar1"] {
        let p = scored(prep(F1, F1_DATA, None, f1_corr(kind, "0.6", "0.0")));
        assert!(p.parsed.model.residual_serial.is_some());
        let (_, s) = solve_at(&p);
        eprintln!(
            "N1 {kind}: {} steps, criterion {:.3e}, OFV {:.10}",
            s.iterations, s.criterion, s.ofv
        );
        assert_eq!(
            s.iterations, 1,
            "{kind}: the Newton matrix is not the correlated objective's Hessian in the \
             levels (criterion {:.3e})",
            s.criterion
        );
        assert!(s.criterion.is_finite() && s.criterion <= PROFILE_TOL);
    }
}

/// The unprofiled fit's own scoring of `params`: a cold inner loop, then `pop_nll_opts`.
fn unprofiled_ofv(p: &crate::api::PreparedRun, params: &ModelParameters) -> f64 {
    let o = options(p);
    let model = &p.parsed.model;
    let mu = compute_mu_k(model, &params.theta, o.mu_referencing);
    let (e, h, _, k) = run_inner_loop_warm_seeded(
        model,
        &p.population,
        params,
        o.inner_maxiter,
        PROFILE_INNER_TOL,
        None,
        Some(&mu),
        0,
        o.inner_restarts,
        InnerHessianSeed::for_options(&o),
        &o,
    );
    2.0 * crate::estimation::outer_optimizer::pop_nll_opts(
        model,
        &p.population,
        params,
        &e,
        &h,
        &k,
        &o,
    )
}

/// D0 with the line: the solve's objective is the unprofiled fit's own scoring
/// (`pop_nll_opts` after a cold inner loop) at the point the solve returns, on F1 with
/// `car1` and `ar1` and on the nested fixture. Dies when the solve scores its subjects
/// without the line (a diagonal `R` in `ProfileCtx::resolve` / `held`): the solve then
/// minimises — and reports — another objective.
#[test]
fn the_correlated_profiled_objective_is_the_unprofiled_objective_at_its_own_point() {
    let cases: Vec<(&str, crate::api::PreparedRun)> = vec![
        (
            "F1 car1",
            scored(prep(F1, F1_DATA, None, f1_corr("car1", "0.6", "0.0"))),
        ),
        (
            "F1 ar1",
            scored(prep(F1, F1_DATA, None, f1_corr("ar1", "0.6", "-0.95"))),
        ),
        (
            "nested ar1",
            scored(prep(NESTED, NESTED_DATA, None, nested_corr)),
        ),
    ];
    for (name, p) in &cases {
        let (_, s) = solve_at(p);
        let plain = unprofiled_ofv(p, &s.params);
        eprintln!(
            "D0 {name}: profiled {:.12} plain {:.12} (Δ {:.2e}), {} steps",
            s.ofv,
            plain,
            s.ofv - plain,
            s.iterations
        );
        assert!(s.ofv.is_finite() && plain.is_finite(), "{name}");
        assert!(
            (s.ofv - plain).abs() < D0_BOUND,
            "{name}: the profile solve reports {} where the unprofiled objective at its own \
             point is {}: the solve's C̃ is not the objective's",
            s.ofv,
            plain
        );
    }
}

/// ρ = 0 is the model without the line: the profiled F1 solve with `ar1(RHO)` at a
/// free `RHO = 0` returns the OFV and levels of plain F1, and ρ ≠ 0 moves the OFV — the
/// straddle, so the identity is not satisfied by a solve that ignores the line.
///
/// Not bit for bit, and not because of the solve: a line at ρ = 0 writes no off-diagonal
/// into `R`, but `ResidualCov::is_dense` is structural, so the subject is scored by the
/// dense-`R` marginal, whose Cholesky sums in another order than the scalar one. The
/// objective itself — the unprofiled `pop_nll_opts`, no profile in it — differs by the
/// same few ULP at one point, and that gap is measured here next to the profile's.
#[test]
fn a_profiled_correlation_at_rho_zero_is_the_model_without_it() {
    let p_plain = scored(prep(F1, F1_DATA, None, |t| t));
    let p_zero = scored(prep(F1, F1_DATA, None, f1_corr("ar1", "0.0", "-0.95")));
    let (_, plain) = solve_at(&p_plain);
    let (_, zero) = solve_at(&p_zero);
    let (_, live) = solve_at(&scored(prep(
        F1,
        F1_DATA,
        None,
        f1_corr("ar1", "0.6", "-0.95"),
    )));
    // RHO is declared after the block, so the level θ keep their indices.
    let n = plain.params.theta.len();
    assert_eq!(zero.params.theta.len(), n + 1);
    assert_eq!(zero.params.theta[n], 0.0);
    // The objective's own gap, at plain F1's solved point with ρ = 0 appended.
    let mut at = plain.params.clone();
    at.theta.push(0.0);
    let objective_gap = unprofiled_ofv(&p_zero, &at) - unprofiled_ofv(&p_plain, &plain.params);
    let profile_gap = zero.ofv - plain.ofv;
    let worst_level = (0..n)
        .map(|k| (zero.params.theta[k] - plain.params.theta[k]).abs())
        .fold(0.0_f64, |a, v| {
            assert!(v.is_finite());
            a.max(v)
        });
    eprintln!(
        "ρ = 0: plain {:.15}, ρ = 0 {:.15} (Δ {profile_gap:.3e}; the objective alone \
         {objective_gap:.3e}), worst level {worst_level:.3e}; ρ = 0.6 {:.12}",
        plain.ofv, zero.ofv, live.ofv
    );
    assert!(objective_gap.is_finite() && profile_gap.is_finite());
    assert!(
        profile_gap.abs() <= RHO_ZERO_OFV,
        "the profiled ρ = 0 fit is {profile_gap:.3e} from the model without the line"
    );
    assert!(objective_gap.abs() <= RHO_ZERO_OFV, "{objective_gap:.3e}");
    assert!(worst_level <= RHO_ZERO_LEVEL, "{worst_level:.3e}");
    assert!(live.ofv.is_finite());
    assert!(
        (live.ofv - plain.ofv).abs() > RHO_MOVES,
        "ρ = 0.6 moved the OFV by only {:.3e}",
        live.ofv - plain.ofv
    );
}

/// Realised (macOS arm64): the profiled ρ = 0 fit 2.8e-14 from plain F1, the objective
/// alone 8.5e-14 at one point — a few ULP of an OFV of 157. 1e-12 is 12× the larger.
const RHO_ZERO_OFV: f64 = 1e-12;
/// Worst realised level gap 1.3e-12 (the solve's FD differences of two objectives a few
/// ULP apart); 75× headroom.
const RHO_ZERO_LEVEL: f64 = 1e-10;

/// The smallest OFV move ρ = 0.6 must make on F1: realised 16.25 (157.41 → 141.15).
const RHO_MOVES: f64 = 1.0;
