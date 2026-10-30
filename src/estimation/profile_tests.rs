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
use crate::estimation::inner_optimizer::run_inner_loop_warm_seeded;
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
    assert!(
        s.n_adjoint_gradients > 0 && s.fd_gradient_subjects.is_empty(),
        "N1 runs on the adjoint level gradient (#2109)"
    );
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

// ── #2114: the affine forms the route check collects ───────────────────────────────

/// F1 with its block read through `read`, a `[individual_parameters]` body that sets
/// `PL`, and the profile solve's objective and step count there.
fn f1_read_as(read: &str) -> (f64, usize) {
    let p = prep(F1, F1_DATA, None, |t| {
        t.replace(
            "  BASE = TVE0 + PLACEBO + ETA_E0 + KAPPA_ARM",
            &format!("{read}\n  BASE = TVE0 + PL + ETA_E0 + KAPPA_ARM"),
        )
    });
    let layout = layout_of(&p);
    let o = options(&p);
    let c = ctx(&p, &layout, &o);
    let s = c
        .solve(
            &p.init_params,
            &ProfileWarm::from_template(&layout, &p.init_params),
        )
        .unwrap_or_else(|e| panic!("{read}: {e:?}"));
    assert!(s.ofv.is_finite(), "{read}: {}", s.ofv);
    (s.ofv, s.iterations)
}

/// Each form #2114 admits is the same model as the one-read rewrite the route check
/// already accepted, `PLACEBO × c` with `c` computed from the data: the solve lands on
/// the same objective, and in one Newton step, which it takes only when the prediction
/// is affine in the levels (N1). The reference is not the plain F1 read, whose
/// objective differs from every row here by far more than the bound (the straddle).
#[test]
fn a_collected_affine_form_profiles_like_its_one_read_rewrite() {
    let (plain, _) = f1_read_as("  PL = PLACEBO");
    for (form, rewrite) in [
        (
            "  if (DOSE > 10) {\n    PL = PLACEBO * 2\n  } else {\n    PL = PLACEBO\n  }",
            "  C = 1\n  if (DOSE > 10) {\n    C = 2\n  }\n  PL = PLACEBO * C",
        ),
        (
            "  PL = if (DOSE > 10) PLACEBO * 2 else PLACEBO",
            "  PL = PLACEBO * (if (DOSE > 10) 2 else 1)",
        ),
        (
            "  A = PLACEBO * DOSE / 100\n  B = PLACEBO\n  PL = A + B",
            "  PL = PLACEBO * (DOSE / 100 + 1)",
        ),
    ] {
        let (want, n_want) = f1_read_as(rewrite);
        let (got, n_got) = f1_read_as(form);
        eprintln!(
            "#2114 {form:?}: {got:.10} vs rewrite {want:.10} (Δ {:.2e}), plain read Δ {:.3e}",
            got - want,
            plain - want
        );
        assert_eq!((n_got, n_want), (1, 1), "{form}: one Newton step each");
        assert!(
            (got - want).abs() < COLLECTED_OFV_BOUND,
            "{form}: {got} vs {want}"
        );
        assert!(
            (plain - want).abs() > 1e3 * COLLECTED_OFV_BOUND,
            "{form}: the straddle, {plain} vs {want}"
        );
    }
}

/// Worst realised |ΔOFV| 8.5e-14 (the sum of reads; both `if` forms are bit-identical
/// to their rewrite), macOS arm64: the two spellings differ only in where the rounding
/// of `c × β` falls. ~1000× headroom; the plain read is 19.3 OFV away.
const COLLECTED_OFV_BOUND: f64 = 1e-10;

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
/// value instead of `β̂`. Returns how many levels the solve holds on a bound (#2112).
fn envelope_matches_fd(edit: impl Fn(String) -> String, analytic: bool) -> usize {
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
    // The solve's level gradient runs on the outer gradient's engine (#2109).
    assert_eq!(s.n_adjoint_gradients > 0, analytic, "the solve's engine");
    assert!(s.fd_gradient_subjects.is_empty());
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
    layout.at_bound(&s.params.theta).len()
}

/// G1b (#2112): with two levels held on their bound, the envelope gradient — the
/// production gradient at `β̂`, unchanged code — is still the gradient of the (bounded)
/// profile objective, on the reconverged FD engine. Danskin: the box does not depend on θ.
/// Realised (Linux aarch64): envelope identity 1.29e-5, handed-vs-profile 1.89e-5 (FD
/// engine); 3.5e-6 and 6.0e-5 (analytic). G1's bands. Dies when the box is dropped from
/// the layout (no level pinned) or the step is not the box QP's.
#[test]
fn the_envelope_gradient_is_the_bounded_profile_gradient_on_the_fd_engine() {
    let pinned = envelope_matches_fd(
        |t| {
            boxed("(0.0, -10.0, 1.0)")(t).replace(
                "  method     = focei\n",
                "  method     = focei\n  gradient   = fd\n",
            )
        },
        false,
    );
    assert_eq!(pinned, 2, "the point has levels on their bound");
}

/// G1b on the analytic engine (no kappa).
#[test]
fn the_envelope_gradient_is_the_bounded_profile_gradient_on_the_analytic_engine() {
    let pinned = envelope_matches_fd(
        |t| {
            boxed("(0.0, -10.0, 1.0)")(t)
                .replace("  kappa KAPPA_ARM ~ 1.0\n", "")
                .replace(" + KAPPA_ARM", "")
                .replace("  iov_column = OCC\n", "")
        },
        true,
    );
    assert!(pinned >= 1, "the point has a level on its bound");
}

/// The envelope identity `dO_p/dθ = ∂O/∂θ|β̂`, both sides FD, no engine involved. Worst
/// realised 3.1e-5 (F2, three studies, Linux aarch64 and macOS arm64), the β residual at `PROFILE_TOL` and
/// FD noise; ~3× headroom.
const G1_IDENTITY_BOUND: f64 = 1e-4;
/// The gradient the optimizer is handed against FD of the profile objective. This also
/// carries the production gradient's own distance from its objective, which is the larger
/// part: on the analytic arm `∂O/∂TVE0` reads 25.58695 against an FD of 25.59225 at held
/// `β̂` (2.0e-4 relative), a property of the unprofiled engine, not of the profile. Worst
/// realised at the `1e-7` stencil tolerance (Linux aarch64, #2378): 1.78e-4 (analytic),
/// 3.1e-5 (FD, IOV) and 2.1e-5 (FD, no IOV) — the FD arms were 5.5e-5 before #2374 — so
/// ~3× headroom over the analytic arm. The `car1` arm (#2110): 1.3e-5, identity 2.9e-5
/// (macOS arm64); the solve on a diagonal `R` against the dense production gradient puts
/// it at 0.87.
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
                // `TVE0` then reads nothing, which the level binder's rank
                // check refuses (#2421): fixed.
                t.replace("BASE  = TVE0 + PLACEBO", "BASE  = PLACEBO")
                    .replace("theta TVE0(45.0, 0.0, 200.0)", "theta TVE0(45.0, FIX)")
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
    .expect("F1 profiles a level block")
    .0;
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

/// #2214: the covariance step's re-profiling solve goes through the optimizer's EBE guard,
/// on the replay entry and inline alike (`covariance_at_estimate`, shared). A starved inner
/// solve (`inner_maxiter = 0` from cold EBEs: both of the 2-study F1's subjects
/// unconverged, measured; none rejected at their start) under `max_unconverged_frac = 0` fails the
/// step with the guard's warning, counts named, classified `covariance_failed`, its time
/// recorded. The same call at `max_unconverged_frac = 1` passes the guard: both sides of
/// the gate in one test, so a guard stuck either way fails here.
#[test]
fn a_guarded_covariance_resolve_fails_the_step() {
    let p = prep(F1, F1_DATA, Some(2), |t| t);
    let n = p.population.subjects.len();
    let run = |frac: f64| {
        let o = FitOptions {
            covariance_method: CovarianceMethod::Hessian,
            inner_maxiter: 0,
            max_unconverged_frac: frac,
            ..options(&p)
        };
        replay_covariance(
            &p.parsed.model,
            &p.population,
            &p.init_params,
            &pack_with_bounds(&p.init_params).packed,
            &[],
            &[],
            &o,
        )
        .expect("F1 profiles a level block")
        .0
    };
    assert_eq!(n, 2);
    let guarded = run(0.0);
    assert!(guarded.matrix.is_none());
    assert_eq!(guarded.warnings.len(), 1, "{:?}", guarded.warnings);
    let w = &guarded.warnings[0];
    assert!(
        w.starts_with(&format!(
            "Covariance step failed: the profile solve at the estimates left 2 of {n} \
             subjects' EBEs unconverged and rejected 0 at their inner start."
        )),
        "{w}"
    );
    assert!(
        w.contains(
            " The EBE guard rejects an unconverged fraction above max_unconverged_frac = 0."
        ),
        "{w}"
    );
    assert!(w.ends_with(" SE estimates not available."), "{w}");
    let c = crate::types::classify_warning(w);
    assert_eq!(
        (c.severity, c.category.as_str()),
        (crate::types::WarningSeverity::Critical, "covariance_failed"),
        "{w}"
    );
    assert!(guarded.wall_time_secs > 0.0);

    let passed = run(1.0);
    assert!(
        !passed.warnings.iter().any(|w| w.contains("EBE guard")),
        "{:?}",
        passed.warnings
    );
}

/// The guard's warning names the trigger that fired, over both cells of its input space
/// (#2388 review row 1): a start rejection, which `ebe_guard_rejects` rejects whatever
/// `max_unconverged_frac` is (here 1, a guard on the fraction that can never fire, with no
/// subject unconverged), and an unconverged fraction above it. Each cell's reason is
/// asserted present and the other's absent, so forcing the branch either way fails here.
#[test]
fn the_guarded_resolve_warning_names_the_trigger_that_fired() {
    let stats = |n_unconverged, n_start_rejected| InnerLoopStats {
        n_unconverged,
        n_fallback: 0,
        n_start_rejected,
    };
    let start = profile_resolve_guarded_msg(&stats(0, 1), 12, 1.0, false);
    assert_eq!(
        start,
        "Covariance step failed: the profile solve at the estimates left 0 of 12 subjects' \
         EBEs unconverged and rejected 1 at their inner start. The EBE guard rejects any \
         subject rejected at its inner start, whatever max_unconverged_frac is. SE estimates \
         not available."
    );
    let frac = profile_resolve_guarded_msg(&stats(3, 0), 12, 0.1, false);
    assert_eq!(
        frac,
        "Covariance step failed: the profile solve at the estimates left 3 of 12 subjects' \
         EBEs unconverged and rejected 0 at their inner start. The EBE guard rejects an \
         unconverged fraction above max_unconverged_frac = 0.1. SE estimates not available."
    );
    assert!(!start.contains("max_unconverged_frac = "), "{start}");
    assert!(!frac.contains("whatever"), "{frac}");
}

/// #2214: a covariance step whose re-profiling solve fails, reached through `fit()` (the
/// `FAIL_COVARIANCE_RESOLVE` θ name: the optimizer's own solve has no natural way to succeed
/// while the same solve from its mode fails). The fit keeps its estimates and reports the
/// step failed with `profile_solve_failed_msg`'s text, and the time it spent, not `0.0`.
#[test]
fn a_failed_inline_resolve_fails_the_fit_covariance_step() {
    let p = prep(F1, F1_DATA, Some(2), |t| {
        t.replace("TVE0", FAIL_COVARIANCE_RESOLVE)
    });
    let o = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: true,
        covariance_method: CovarianceMethod::Hessian,
        ..options(&p)
    };
    let f = crate::api::fit(&p.parsed.model, &p.population, &p.init_params, &o)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(f.profiled_blocks, ["PLACEBO"]);
    assert!(f.ofv.is_finite(), "the estimates stand: {}", f.ofv);
    assert!(f.covariance_matrix.is_none());
    assert_eq!(f.covariance_status, crate::types::CovarianceStatus::Failed);
    let msg = "Covariance step failed: the profile solve did not converge at the estimates \
               (Singular). SE estimates not available.";
    assert!(
        f.warnings.iter().any(|w| w.contains(msg)),
        "{:?}",
        f.warnings
    );
    assert!(f.covariance_wall_time_secs > 0.0);
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
/// optimum, #2285 M5); not a spent budget, a cancelled run or a bad argument. BOBYQA's
/// are its three tolerance stops, the #1782 polish set, from the start too (#2336).
/// A `Failure` / `RoundoffLimited` that has not left the start is not verified (#2338
/// review #3: a first step that fails would otherwise be verified by a leg that gains
/// nothing); a tolerance stop is verified either way. Dies when the `left_init` gate is
/// dropped, or applied to tolerance stops too; and when BOBYQA is excluded again, or
/// verified on a stop that is not a tolerance one.
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
    for r in [ok(S::Success), ok(S::FtolReached), ok(S::XtolReached)] {
        for left_init in [true, false] {
            assert!(
                profiled_stop_needs_verify(bobyqa, &r, false, left_init),
                "BOBYQA {r:?}, left_init {left_init}"
            );
        }
        assert!(
            !profiled_stop_needs_verify(bobyqa, &r, true, true),
            "cancelled BOBYQA {r:?}"
        );
    }
    for r in [
        ok(S::StopValReached),
        ok(S::MaxEvalReached),
        ok(S::MaxTimeReached),
        err(F::Failure),
        err(F::RoundoffLimited),
        err(F::ForcedStop),
    ] {
        assert!(
            !profiled_stop_needs_verify(bobyqa, &r, false, true),
            "BOBYQA {r:?}"
        );
    }
}

/// The verdict comes from the verification gain, not from NLopt's status: a settled
/// leg is converged on `Failure` as on `XtolReached`; a leg cap, a spent budget or a leg
/// that could not start is not; with nothing verified the status decides. BOBYQA is
/// judged the same way on its polish legs (#2336). Dies when `Failure` is mapped to
/// unconverged again.
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
    let verdict = |end, r: NloptReturn| profiled_converged(bobyqa, end, &r, false, true);
    assert!(verdict(PolishEnd::Settled, ok(S::FtolReached)));
    assert!(verdict(PolishEnd::Settled, ok(S::XtolReached)));
    assert!(!verdict(PolishEnd::Settled, ok(S::MaxEvalReached)));
    assert!(!verdict(
        PolishEnd::LegCapReached { last_gain: 2e-5 },
        ok(S::FtolReached)
    ));
    assert!(!verdict(PolishEnd::BudgetSpent, ok(S::FtolReached)));
    assert!(!verdict(PolishEnd::NotNeeded, err(F::Failure)));
}

/// #2336: a profiled BOBYQA fit's legs run at the #1782 `POLISH_*` ceilings, a tighter
/// user value winning; a gradient method's legs keep its first leg's stops. Dies when
/// BOBYQA's legs restart at its first-leg tolerances (they then stop where it did), or
/// the ceilings reach L-BFGS / SLSQP.
#[test]
fn a_profiled_bobyqa_leg_polishes_at_the_polish_tolerances() {
    let p = prep(F2, F2_DATA, Some(2), |t| t);
    let model = &p.parsed.model;
    let o = options(&p);
    let bobyqa = nlopt::Algorithm::Bobyqa;
    assert!(POLISH_XTOL_REL < profiled_xtol(bobyqa, &o));
    assert!(POLISH_FTOL_REL < profiled_ftol(bobyqa, model, &o));
    assert_eq!(
        profiled_leg_tols(bobyqa, model, &o),
        (POLISH_XTOL_REL, POLISH_FTOL_REL)
    );
    let tighter = FitOptions {
        outer_xtol: 1e-9,
        outer_ftol: Some(1e-12),
        ..o.clone()
    };
    assert_eq!(profiled_leg_tols(bobyqa, model, &tighter), (1e-9, 1e-12));
    for algo in [nlopt::Algorithm::Lbfgs, nlopt::Algorithm::Slsqp] {
        assert_eq!(
            profiled_leg_tols(algo, model, &o),
            (profiled_xtol(algo, &o), profiled_ftol(algo, model, &o))
        );
    }
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
    assert!(!cap.contains("still moving"), "{cap:?}");

    // #2340: a cap reached by creeping legs, whose last gain is under the threshold, says
    // so rather than "still improving" by a gain that reads like a converged fit (#2465
    // review #2). Both sides of the gate are asserted here, so a variant stuck on either
    // branch fails.
    let creep = w(
        PolishEnd::LegCapReached {
            last_gain: CREEPER_GAIN,
        },
        ok(S::XtolReached),
        false,
    )
    .expect("a creeping leg cap warns");
    for sentence in [
        format!("still moving after {MAX_POLISH_LEGS} verification legs"),
        "the last stopped on its step tolerance while still moving".to_string(),
        "lowering the objective by 7.390e-8".to_string(),
        "so it is not reported as converged".to_string(),
        "re-run from these estimates".to_string(),
    ] {
        assert!(
            creep.contains(&sentence),
            "{sentence:?} missing from {creep:?}"
        );
    }
    assert!(!creep.contains("still improving"), "{creep:?}");

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

/// A scripted verification leg at the default profiled `xtol_rel`.
fn scripted_leg(ret: NloptReturn, gain: f64, displacement: f64) -> LegOutcome {
    LegOutcome {
        ret,
        gain,
        displacement,
        xtol: PROFILED_GRADIENT_XTOL_REL,
    }
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
    let mut script = vec![
        scripted_leg(ok(S::XtolReached), 2.1e-5, 0.0),
        scripted_leg(err(F::Failure), 6e-8, 0.0),
    ]
    .into_iter();
    let (last, end) = profiled_verify_legs(
        lbfgs,
        err(F::Failure),
        |r| profiled_stop_needs_verify(lbfgs, r, false, true),
        || 100,
        |_| script.next(),
    );
    assert_eq!(end, PolishEnd::Settled);
    assert!(script.next().is_none(), "both legs ran");
    assert!(profiled_converged(lbfgs, end, &last, false, true));

    let (last, end) = profiled_verify_legs(
        lbfgs,
        ok(S::FtolReached),
        |r| profiled_stop_needs_verify(lbfgs, r, false, true),
        || 100,
        |_| Some(scripted_leg(ok(S::XtolReached), 2.1e-5, 0.0)),
    );
    assert_eq!(end, PolishEnd::LegCapReached { last_gain: 2.1e-5 });
    assert!(!profiled_converged(lbfgs, end, &last, false, true));

    // A `Failure` that never left the start runs no leg and is not converged.
    let (last, end) = profiled_verify_legs(
        lbfgs,
        err(F::Failure),
        |r| profiled_stop_needs_verify(lbfgs, r, false, false),
        || 100,
        |_| -> Option<LegOutcome> { panic!("no leg may run from a failed start") },
    );
    assert_eq!(end, PolishEnd::NotNeeded);
    assert!(!profiled_converged(lbfgs, end, &last, false, false));
}

/// #2340's legs, measured on A2 (Linux aarch64, module doc of [`PROFILED_LEG_CREEP`]):
/// the leg from (50, 6, 2) that stopped `XtolReached` after a 3.67e-7 displacement,
/// gaining 7.39e-8 while 1.24e-5 short; the largest displacement of an `XtolReached` leg
/// that settles on the optimum, 7.9e-8; and S1's settling `FtolReached` leg, 1.3e-5 at a
/// gain of 9.4e-7.
const CREEPER_DISPLACEMENT: f64 = 3.67e-7;
const CREEPER_GAIN: f64 = 7.39e-8;
const GENUINE_XTOL_DISPLACEMENT: f64 = 7.9e-8;
const S1_FTOL_DISPLACEMENT: f64 = 1.3e-5;
const S1_FTOL_GAIN: f64 = 9.4e-7;

/// #2340, a differential pair across the creep gate: two `XtolReached` legs with the
/// same sub-threshold gain, differing only in displacement, on either side of
/// [`PROFILED_LEG_CREEP`] × `xtol_rel`. The creeper does not settle and the genuine leg
/// does. The straddle itself is asserted, so the pair cannot quietly fall on one side.
/// Around them: an `FtolReached` leg settles however far it moved (S1), BOBYQA keeps the
/// gain alone, SLSQP is judged like L-BFGS, a gain over [`PROFILED_VERIFY_GAIN`] never
/// settles, and a `NaN` gain passes the gain test, as in `drive_polish_legs`, but not
/// the creep test.
///
/// Dies when the displacement test is dropped (the creeper settles), when it fires on
/// every `XtolReached` leg or compares against `xtol` itself (the genuine leg does not
/// settle), when it covers `FtolReached` (S1 does not settle), when it reaches BOBYQA, or
/// when a `NaN` displacement settles (`displacement > gate` rather than `!(… <= gate)`).
#[test]
fn a_leg_that_stopped_on_its_step_test_while_moving_does_not_settle() {
    use nlopt::{Algorithm as A, SuccessState as S};
    let xtol = PROFILED_GRADIENT_XTOL_REL;
    let gate = PROFILED_LEG_CREEP * xtol;
    assert!(
        GENUINE_XTOL_DISPLACEMENT <= gate && gate < CREEPER_DISPLACEMENT,
        "the pair no longer straddles the gate {gate:e}"
    );
    assert!(
        CREEPER_GAIN <= PROFILED_VERIFY_GAIN,
        "the creeper's gain alone settles"
    );
    let settles = |algo, status, gain, displacement| {
        profiled_leg_settles(algo, &ok(status), gain, displacement, xtol)
    };
    for algo in [A::Lbfgs, A::Slsqp] {
        assert!(
            !settles(algo, S::XtolReached, CREEPER_GAIN, CREEPER_DISPLACEMENT),
            "{algo:?}: the creeping leg settled"
        );
        assert!(
            settles(
                algo,
                S::XtolReached,
                CREEPER_GAIN,
                GENUINE_XTOL_DISPLACEMENT
            ),
            "{algo:?}: a genuine optimum's leg did not settle"
        );
        assert!(
            settles(algo, S::FtolReached, S1_FTOL_GAIN, S1_FTOL_DISPLACEMENT),
            "{algo:?}: S1's FtolReached leg did not settle"
        );
        assert!(!settles(
            algo,
            S::XtolReached,
            2.0 * PROFILED_VERIFY_GAIN,
            0.0
        ));
        assert!(settles(algo, S::XtolReached, f64::NAN, 0.0));
        // #2465 review #8: a `NaN` gain passes the gain test only. A creeping leg with one
        // still runs on, and a `NaN` displacement does not settle either.
        assert!(!settles(
            algo,
            S::XtolReached,
            f64::NAN,
            CREEPER_DISPLACEMENT
        ));
        assert!(!settles(algo, S::XtolReached, CREEPER_GAIN, f64::NAN));
    }
    assert!(
        settles(
            A::Bobyqa,
            S::XtolReached,
            CREEPER_GAIN,
            CREEPER_DISPLACEMENT
        ),
        "BOBYQA's polish legs are judged on the gain alone"
    );
    // Relative to the leg's own `xtol`: a tighter user `outer_xtol` tightens the gate.
    assert!(!profiled_leg_settles(
        A::Lbfgs,
        &ok(S::XtolReached),
        CREEPER_GAIN,
        GENUINE_XTOL_DISPLACEMENT,
        xtol / 10.0
    ));
}

/// [`leg_displacement`]: the largest move of a free coordinate over the largest free
/// coordinate floored at 1, ignoring held coordinates, and `NaN` when any free move is
/// (#2465 review #8). Dies when the fold is `f64::max` (the `NaN` is dropped and the move
/// reads 0.5), when the floor or the held mask is lost, or when the scale is taken from
/// `to` instead of `from`.
#[test]
fn a_leg_displacement_is_the_largest_relative_move_and_keeps_nan() {
    let from = [4.0, -0.5, 100.0];
    let to = [4.5, -0.25, 300.0];
    // Coordinate 2 is held: its move and its magnitude are not read.
    assert_eq!(leg_displacement(&from, &to, &[0, 1]), 0.5 / 4.0);
    assert_eq!(leg_displacement(&from, &from, &[0, 1]), 0.0);
    // Below 1 the scale is floored at 1.
    assert_eq!(leg_displacement(&[0.5, 0.25], &[0.75, 0.25], &[0, 1]), 0.25);
    // The scale is `from`'s, not `to`'s.
    assert_eq!(leg_displacement(&[2.0], &[8.0], &[0]), 3.0);
    assert!(leg_displacement(&[4.0, 1.0], &[f64::NAN, 1.5], &[0, 1]).is_nan());
    assert!(leg_displacement(&[4.0, 1.0], &[4.5, f64::NAN], &[0, 1]).is_nan());
}

/// #2340 through the loop, on (50, 6, 2)'s measured legs: the creeping first leg does
/// not settle, so the loop runs on, its second leg gains 1.23e-5, and the third settles,
/// converged. Four creeping legs in a row reach the leg cap and are not converged, with
/// the leg-cap warning. Dies when the loop settles on the creeper (one leg runs and the
/// fit is converged 1.24e-5 short), or when `profiled_verify_legs` hands the settle test
/// some other displacement or `xtol` than the leg's.
#[test]
fn a_creeping_leg_is_followed_by_another() {
    use nlopt::SuccessState as S;
    let lbfgs = nlopt::Algorithm::Lbfgs;
    let mut script = vec![
        scripted_leg(ok(S::XtolReached), CREEPER_GAIN, CREEPER_DISPLACEMENT),
        scripted_leg(ok(S::FtolReached), 1.23e-5, 3.0e-4),
        scripted_leg(ok(S::XtolReached), 2.5e-10, 2.9e-12),
    ]
    .into_iter();
    let mut ran = 0;
    let (last, end) = profiled_verify_legs(
        lbfgs,
        ok(S::FtolReached),
        |r| profiled_stop_needs_verify(lbfgs, r, false, true),
        || 100,
        |_| {
            ran += 1;
            script.next()
        },
    );
    assert_eq!(
        (ran, end),
        (3, PolishEnd::Settled),
        "the loop stopped on the creeper"
    );
    assert!(profiled_converged(lbfgs, end, &last, false, true));
    assert_eq!(profiled_verdict_warning(end, &last, true), None);

    let (last, end) = profiled_verify_legs(
        lbfgs,
        ok(S::FtolReached),
        |r| profiled_stop_needs_verify(lbfgs, r, false, true),
        || 100,
        |_| {
            Some(scripted_leg(
                ok(S::XtolReached),
                CREEPER_GAIN,
                CREEPER_DISPLACEMENT,
            ))
        },
    );
    assert_eq!(
        end,
        PolishEnd::LegCapReached {
            last_gain: CREEPER_GAIN
        }
    );
    assert!(!profiled_converged(lbfgs, end, &last, false, true));
    let w = profiled_verdict_warning(end, &last, false).expect("a leg cap warns");
    assert!(
        w.contains("while still moving"),
        "the creep cap's reason: {w}"
    );

    // The gate reads each leg's own `xtol` (#2465 review #1): a genuine-sized displacement
    // on a leg run at a tenth of the default `xtol_rel` is creeping, so the loop runs on.
    let mut ran = 0;
    let (_, end) = profiled_verify_legs(
        lbfgs,
        ok(S::FtolReached),
        |r| profiled_stop_needs_verify(lbfgs, r, false, true),
        || 100,
        |_| {
            ran += 1;
            Some(LegOutcome {
                xtol: PROFILED_GRADIENT_XTOL_REL / 10.0,
                ..scripted_leg(ok(S::XtolReached), CREEPER_GAIN, GENUINE_XTOL_DISPLACEMENT)
            })
        },
    );
    assert_eq!(
        (ran, end),
        (
            MAX_POLISH_LEGS,
            PolishEnd::LegCapReached {
                last_gain: CREEPER_GAIN
            }
        ),
        "the loop judged the leg at another xtol than its own"
    );
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
        // #2340: the displacement the settle test reads is the incumbent's move. A move
        // means an improvement, and an improvement beyond the profile solve's noise means
        // a move. A gain within noise at the same point is a warm re-solve scoring lower,
        // not a move (#2465 review #9).
        assert!(l.displacement.is_finite(), "leg {k}: {l:?}");
        if l.displacement > 0.0 {
            assert!(l.gain > 0.0, "leg {k} moved without improving: {l:?}");
        }
        if l.gain > OFV_NOISE * l.before.abs().max(1.0) {
            assert!(
                l.displacement > 0.0,
                "leg {k} reported a displacement it did not make: {l:?}"
            );
        }
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

/// #2336: a profiled BOBYQA fit polishes its tolerance stop: F2 on its first two studies,
/// through [`optimize_profiled`], every leg at the `POLISH_*` tolerances read back from
/// the leg itself and reporting its real gain, ending on the optimum the default L-BFGS
/// fit reaches. The verdict follows the last gain: converged when it is no more than
/// [`PROFILED_VERIFY_GAIN`], otherwise the leg cap, not converged, with its warning.
/// Dies when BOBYQA is not verified (no leg runs and the first stop is reported, more
/// than `BOBYQA_FIRST_STOP_SHORT` above that optimum), or its legs run at the first
/// leg's tolerances.
///
/// Measured (macOS arm64): the first stop 6.45e-2 above L-BFGS's 22.139638204; four
/// legs gaining 6.4e-2, 1.5e-4, 4.6e-5, 4.3e-6 end 3.5e-8 above it on the leg cap — the
/// last gain is over the threshold, so the fit says not converged, which #2336 accepts.
#[test]
fn a_profiled_bobyqa_fit_polishes_its_tolerance_stop() {
    let p = prep(F2, F2_DATA, Some(2), |t| t);
    let base = FitOptions {
        run_covariance_step: false,
        ..options(&p)
    };
    let lbfgs = optimize_profiled(&p.parsed.model, &p.population, &p.init_params, &base);
    assert!(lbfgs.converged, "{:?}", lbfgs.warnings);
    let o = FitOptions {
        optimizer: Optimizer::Bobyqa,
        ..base
    };
    take_verify_legs();
    let r = optimize_profiled(&p.parsed.model, &p.population, &p.init_params, &o);
    let legs = take_verify_legs();
    eprintln!(
        "F2 BOBYQA legs: {legs:?}; OFV {:.9} vs L-BFGS {:.9} (Δ {:.3e})",
        r.ofv,
        lbfgs.ofv,
        r.ofv - lbfgs.ofv
    );
    assert!(!legs.is_empty(), "no polish leg ran");
    let first_stop = legs[0].before;
    assert!(
        first_stop - lbfgs.ofv > BOBYQA_FIRST_STOP_SHORT,
        "fixture: BOBYQA's first stop {first_stop} is already on the optimum {}",
        lbfgs.ofv
    );
    for (k, l) in legs.iter().enumerate() {
        assert_eq!(l.xtol, Some(POLISH_XTOL_REL), "leg {k}: {l:?}");
        assert_eq!(l.ftol, Some(POLISH_FTOL_REL), "leg {k}: {l:?}");
        assert!(
            l.before.is_finite() && l.after.is_finite(),
            "leg {k}: {l:?}"
        );
        assert_eq!(l.gain, l.before - l.after, "leg {k}: {l:?}");
    }
    let settled = legs.last().unwrap().gain <= PROFILED_VERIFY_GAIN;
    assert_eq!(r.converged, settled, "{legs:?}: {:?}", r.warnings);
    let cap_warned = r.warnings.iter().any(|w| w.contains("verification legs"));
    assert_eq!(cap_warned, !settled, "{:?}", r.warnings);
    if !settled {
        assert_eq!(legs.len(), MAX_POLISH_LEGS, "{legs:?}");
    }
    assert!(r.ofv.is_finite());
    assert!(
        (r.ofv - lbfgs.ofv).abs() < BOBYQA_POLISHED_OFV,
        "BOBYQA {} vs L-BFGS {}",
        r.ofv,
        lbfgs.ofv
    );
}

/// The first stop is 6.45e-2 above the optimum (macOS arm64): 645× this.
const BOBYQA_FIRST_STOP_SHORT: f64 = 1e-4;
/// The polished fit ends 3.5e-8 above it (macOS arm64); `A2_FIT_OFV`'s band.
const BOBYQA_POLISHED_OFV: f64 = 1e-5;

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
    // The level binder refuses a free θ that moves nothing (#2421), so `JUNK`
    // reaches the fit as a model laid out on the levels F2 binds without it —
    // the layout the binder never re-checks — and its default parameters.
    let mut p = prep(F2, F2_DATA, Some(2), |t| t);
    let text = with_junk_theta(std::fs::read_to_string(root().join(F2)).unwrap());
    let mut junk = crate::parser::model_parser::parse_full_model_with(&text, &p.parsed.bindings)
        .unwrap_or_else(|e| panic!("{e}"));
    junk.model.name = p.parsed.model.name.clone();
    p.init_params = junk.model.default_params.clone();
    p.parsed.model = junk.model;
    let o = FitOptions {
        outer_maxiter: 3,
        run_covariance_step: false,
        ..options(&p)
    };
    let r = optimize_profiled(&p.parsed.model, &p.population, &p.init_params, &o);
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
    let pd =
        c.covariance_from_profile_hessian(&hold, &x_hat, &free, &s, &a_pd, Instant::now(), None);
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
    let out =
        c.covariance_from_profile_hessian(&hold, &x_hat, &free, &s, &a_bad, Instant::now(), None);
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
    let LevelBlocks { d_groups, b, .. } = c.level_blocks(&hold, &x_hat, &free, &s).unwrap();
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

/// A profiled fit reports the subjects that declined the analytic outer gradient at
/// runtime, as an unprofiled fit does (#1154): before, `optimize_profiled` built its own
/// decline log and never read it, so the fallback was silent. Non-IOV F2 (analytic at
/// model level), every subject forced to decline: the warning counts them and names the
/// **reconverged** salvage that log gives them, not the held-EBE one. The same fit without
/// the switch is the straddle — no warning. Dies when the warning is not emitted, or is
/// emitted from a plain log's wording.
#[test]
fn a_profiled_fit_reports_its_runtime_outer_fd_fallbacks() {
    let p = prep(F2, F2_DATA, Some(2), |t| {
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
    let n = p.population.subjects.len();
    let o = FitOptions {
        run_covariance_step: false,
        outer_maxiter: 3,
        ..options(&p)
    };
    let fallback = |r: &OuterResult| -> Vec<String> {
        r.warnings
            .iter()
            .filter(|w| w.contains("could not be given the exact analytic outer gradient"))
            .cloned()
            .collect()
    };
    FORCE_PROFILED_DECLINES.with(|c| c.set(true));
    let forced = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        optimize_profiled(&p.parsed.model, &p.population, &p.init_params, &o)
    }));
    FORCE_PROFILED_DECLINES.with(|c| c.set(false));
    let forced = forced.unwrap_or_else(|e| std::panic::resume_unwind(e));
    let w = fallback(&forced);
    assert_eq!(w.len(), 1, "{:?}", forced.warnings);
    assert!(
        w[0].starts_with(&format!("{n} of {n} subjects")),
        "{}",
        w[0]
    );
    assert!(
        w[0].contains("used reconverged finite-difference outer gradients")
            && !w[0].contains("fixed-EBE"),
        "the profiled log reconverges every decline: {}",
        w[0]
    );

    let plain = optimize_profiled(&p.parsed.model, &p.population, &p.init_params, &o);
    assert!(fallback(&plain).is_empty(), "{:?}", plain.warnings);
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
        let step = bordered_step(&g, &m, &gy, &blocks, &[false; 4]).expect("PD blocks");
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
        assert!(bordered_step(&g, &flat, &gy, &blocks, &[false; 4]).is_none());
    }
}

/// The covariance step's `D` and `B`, differenced over each subject's own levels and
/// mapped to θ, equal the dense θ-space FD of the population objective (every EBE
/// re-solved cold) at the solved point, under the global contrast. Dies on a wrong sign
/// or a dropped term in either mapping (`TᵀDT`, `BT`). The blocks run on the adjoint
/// (#2109); C2 pins each subject's against the re-solved differences.
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
    let LevelBlocks { d_groups: d, b, .. } = c.level_blocks(&hold, &x, &free, &s).unwrap();
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
        // Well below the stencil's default: the blocks are exact now (#2109), so the
        // oracle's truncation is the whole gap.
        let floor = fd_step_floor(&hold, &c.model.default_params, free[sidx]);
        let hs = fd_step(LEVEL_BLOCKS_ORACLE_STEP, x[free[sidx]], floor);
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

/// Worst realised gap relative to the largest entry, the blocks exact (#2109) and the
/// oracle at `LEVEL_BLOCKS_ORACLE_STEP`: `D` 2.7e-7, `B` 1.7e-5 (Linux, macOS alike). `B`'s gap is
/// the oracle's truncation: it falls as the step squared, 8.6e-3 / 8.8e-4 / 1.1e-4 / 1.7e-5
/// at 1e-2 / 3e-3 / 1e-3 / 3e-4, toward the exact value. ~3× headroom over `B`. Dropping the
/// border's term from the `B` mapping moved it to 4.1e-1 (measured on the FD blocks).
const LEVEL_BLOCKS_BOUND: f64 = 5e-5;
/// The oracle's structural step.
const LEVEL_BLOCKS_ORACLE_STEP: f64 = 3e-4;

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
/// re-solves run in a fixed order: on the FD route `2k` central-difference probes of
/// `total_gradient_fd`, then the line search's first trial — all from the state the
/// population pass returned — then the second iteration's probes, from the trial's modes.
/// The first `2k + 1` of them must start at the pass's (η̂, κ̂): the gradient's warm start,
/// then the line search's. On the adjoint route (#2109) the gradient re-solves nothing, so
/// the line search's trial is the first.
#[test]
fn profile_gradient_and_line_search_start_at_the_pass_mode() {
    for fd in [true, false] {
        gradient_and_line_search_start_at_the_pass_mode(fd);
    }
}

fn gradient_and_line_search_start_at_the_pass_mode(fd: bool) {
    use crate::estimation::inner_optimizer::record_inner_solves;
    use crate::estimation::inner_optimizer::test_helpers::{
        assert_these_start_at_modes, on_one_thread,
    };
    let p = prep(F1, F1_DATA, Some(2), |t| if fd { forced_fd(t) } else { t });
    assert_eq!(
        profile_gradient_is_analytic(&p.parsed.model, &options(&p)),
        !fd
    );
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
        let probes = if fd { 2 * k } else { 0 };
        assert!(
            mine.len() > probes,
            "subject {}: {} re-solves",
            subject.id,
            mine.len()
        );
        first.extend(mine[..probes].iter().copied());
        line_search.push(mine[probes]);
    }
    if fd {
        assert_these_start_at_modes(
            "profile total_gradient_fd",
            first,
            &p.population,
            &etas,
            &kappas,
        );
    }
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
/// line searches, and the level stencil. On both routes: on the exact one (#2109) the
/// re-profiling passes are all there is, as F1 is solved in its first pass and its blocks
/// are the adjoint's.
#[test]
fn profiled_covariance_starts_at_the_fit_mode() {
    use crate::estimation::inner_optimizer::record_inner_solves;
    use crate::estimation::inner_optimizer::test_helpers::{
        assert_starts_at_modes, assert_starts_at_previous_modes, on_one_thread,
    };
    let p = prep(F1, F1_DATA, Some(2), |t| t);
    let layout = layout_of(&p);
    for exact in [false, true] {
        let o = FitOptions {
            covariance_method: CovarianceMethod::Hessian,
            analytic_cov_hessian: exact,
            ..options(&p)
        };
        let c = ctx(&p, &layout, &o);
        assert_eq!(c.cov_is_analytic(), exact);
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
            "exact={exact}: every re-profiling solve is warm"
        );
        assert_starts_at_modes(
            "profiled covariance re-profile",
            &rec,
            |s| s.pass == Some(0),
            &p.population,
            &fit.etas,
            &fit.kappas,
        );
        if exact {
            // The exact route re-profiles once, at the step's tolerance, and its blocks
            // re-solve nothing: no solve outside that first pass.
            assert!(
                !rec.starts.is_empty() && rec.starts.iter().all(|s| s.pass == Some(0)),
                "the exact route re-solved after the re-profile"
            );
            let tol = c.at_cov_tol().inner_tol;
            assert!(
                rec.solve_tols.iter().all(|&(_, t)| t == tol),
                "a solve off the step's tolerance {tol:e}: {:?}",
                rec.solve_tols
            );
        } else {
            let n = assert_starts_at_previous_modes("profiled covariance stencil", &rec, |s| {
                s.pass != Some(0)
            });
            assert!(
                n > rec.passes.len(),
                "the per-subject re-solves (level stencil, gradient probes) are among the {n}"
            );
        }
    }
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

// ── G2: the adjoint level gradient is the reconverged FD one (#2109) ───────────────

/// Per subject, at the template's levels (away from `β̂`, so the gradient is live), the
/// level gradient the Newton step reads from the adjoint against the reconverged central
/// differences it replaces ([`ProfileCtx::total_gradient_fd`], the EBE re-solved at every
/// point). Both in the solve's own units: a level's gradient times the subject's
/// conditional SE of it, the scale `PROFILE_TOL` is set in. Returns the worst gap and the
/// largest FD gradient (the straddle: a gradient of zero agrees with anything).
fn level_gradient_gaps(p: &crate::api::PreparedRun) -> (f64, f64) {
    let layout = layout_of(p);
    let o = options(p);
    let c = ctx(p, &layout, &o);
    let params = p.init_params.clone();
    let x = pack_params(&params);
    let pt = crate::sens::ode_provider::adjoint::PackedPoint::new(&params, &x).unwrap();
    let zero = DVector::zeros(c.model.n_eta);
    let per: Vec<(f64, f64)> = (0..p.population.subjects.len())
        .into_par_iter()
        .map(|i| {
            let s = c.resolve(i, &params, &zero);
            let k = layout.reads[i].len();
            let m = c.held_hessian(i, &params, &s, &vec![1.0; k]);
            let se: Vec<f64> = (0..k).map(|a| 1.0 / m[(a, a)].sqrt()).collect();
            let steps: Vec<f64> = se.iter().map(|s| FD_STEP_SE * s).collect();
            let fd = c.total_gradient_fd(i, &params, &s, &steps);
            let ad = c
                .total_gradient_adjoint(i, &pt, &s)
                .unwrap_or_else(|| panic!("subject {i}: the adjoint declined"));
            assert_eq!(ad.len(), k);
            let (mut gap, mut size) = (0.0_f64, 0.0_f64);
            for a in 0..k {
                assert!(
                    fd[a].is_finite() && ad[a].is_finite() && se[a].is_finite(),
                    "subject {i} level {a}: fd {} adjoint {} se {}",
                    fd[a],
                    ad[a],
                    se[a]
                );
                gap = gap.max(((ad[a] - fd[a]) * se[a]).abs());
                size = size.max((fd[a] * se[a]).abs());
            }
            (gap, size)
        })
        .collect();
    per.iter()
        .fold((0.0, 0.0), |(g, s), &(gi, si)| (g.max(gi), s.max(si)))
}

/// The fixtures G2 runs on, each with the engine the adjoint must serve it on.
fn g2_fixtures() -> Vec<(&'static str, crate::api::PreparedRun)> {
    vec![
        ("F1", prep(F1, F1_DATA, None, |t| t)),
        ("F2", prep(F2, F2_DATA, Some(3), |t| t)),
        (
            "F2 car1",
            scored(prep(F2, F2_DATA, Some(3), |t| {
                assert!(t.contains("  omega ETA_EM") && t.contains("weight = SE\n"));
                t.replace(
                    "  omega ETA_EM",
                    "  theta RHO(0.6, 0.0, 0.95)\n  omega ETA_EM",
                )
                .replace(
                    "weight = SE\n",
                    "weight = SE\n  correlation car1(RHO) by OCC\n",
                )
            })),
        ),
        (
            "nested",
            scored(prep(NESTED, NESTED_DATA, None, nested_corr)),
        ),
        ("global", global_prep(3)),
    ]
}

/// G2. Dies when the gradient is contracted along a level's own θ only (the slack's
/// `−1` dropped: the global fixture only), when it is taken away from the subject's mode
/// (every nonlinear fixture), and when a subject declines (each one must be served). The
/// engine: every fixture is asserted on the adjoint route. No FOCE arm: under the
/// profile's additive-error scope FOCE and FOCEI are one objective (an F2 `foce` arm
/// measured the same gaps to every printed digit), so the objective choice cannot show.
#[test]
fn the_adjoint_level_gradient_is_the_reconverged_fd_one() {
    for (name, p) in g2_fixtures() {
        assert!(
            profile_gradient_is_analytic(&p.parsed.model, &options(&p)),
            "{name}: the fixture runs on the adjoint"
        );
        if name == "nested" {
            assert!(p.parsed.model.has_nested_levels());
            assert!(
                !crate::sens::ode_provider::adjoint::nested_levels_outside_scope(&p.parsed.model)
            );
        }
        if name == "global" {
            assert!(
                layout_of(&p)
                    .reads
                    .iter()
                    .flatten()
                    .any(|r| r.moves.len() > 1),
                "the global fixture reads a level through a slack"
            );
        }
        let (gap, size) = level_gradient_gaps(&p);
        eprintln!("G2 {name}: worst gap {gap:.3e} SE, largest gradient {size:.3e} SE");
        assert!(
            size > G2_LIVE,
            "{name}: the gradient is not live ({size:.3e})"
        );
        assert!(gap < G2_BOUND, "{name}: {gap:.3e}");
    }
}

/// The adjoint against the reconverged FD gradient, in conditional SEs. Worst realised
/// (Linux aarch64; macOS arm64 the same but nested 7.8e-7): 7.6e-7 (nested), 3.6e-7
/// (global), 1.3e-7 (F2, F2 car1), 9.8e-12 (F1,
/// linear in η, where the central difference is exact): the FD side's noise at
/// `PROFILE_INNER_TOL`, the floor `PROFILE_TOL` was set from. ~4× headroom, and 3× below
/// `PROFILE_TOL` itself.
const G2_BOUND: f64 = 3e-6;
/// The smallest of the fixtures' largest gradients, in SEs, for G2 to mean anything.
/// Realised ≥ 3.25 (global) at the template's levels.
const G2_LIVE: f64 = 1e-2;

// ── R1: the engine the solve's level gradient runs on, and its fallback (#2109) ────

/// F2 with `gradient = fd`, which keeps the whole profile on finite differences.
fn forced_fd(t: String) -> String {
    assert!(t.contains("  method     = focei\n"), "F2 moved");
    t.replace(
        "  method     = focei\n",
        "  method     = focei\n  gradient   = fd\n",
    )
}

/// On F2 every level gradient of the solve comes from the adjoint, none falls back; with
/// `gradient = fd` none comes from the adjoint, and nothing is reported as a fallback
/// (FD is the model's choice there). Dies when the route ignores `gradient = fd`, or when
/// an in-scope fixture silently runs on FD.
#[test]
fn the_profile_gradient_runs_on_the_outer_gradients_engine() {
    for (fd, p) in [
        (false, prep(F2, F2_DATA, Some(3), |t| t)),
        (true, prep(F2, F2_DATA, Some(3), forced_fd)),
    ] {
        assert_eq!(
            profile_gradient_is_analytic(&p.parsed.model, &options(&p)),
            !fd
        );
        let (_, s) = solve_at(&p);
        assert!(
            s.fd_gradient_subjects.is_empty(),
            "fd={fd}: {:?}",
            s.fd_gradient_subjects
        );
        if fd {
            assert_eq!(s.n_adjoint_gradients, 0);
        } else {
            assert!(
                s.n_adjoint_gradients >= p.population.subjects.len(),
                "{} adjoint gradients for {} subjects",
                s.n_adjoint_gradients,
                p.population.subjects.len()
            );
        }
    }
}

/// `f` with subject `i`'s adjoint level gradient declined on this thread.
fn with_declined<T>(i: usize, f: impl FnOnce() -> T) -> T {
    FORCE_PROFILE_GRADIENT_DECLINE.with(|c| c.set(Some(i)));
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    FORCE_PROFILE_GRADIENT_DECLINE.with(|c| c.set(None));
    out.unwrap_or_else(|e| std::panic::resume_unwind(e))
}

/// A subject the adjoint declines takes the reconverged differences and is recorded, and
/// the solve lands where the all-adjoint one does. Dies when a decline is not recorded,
/// or when the fallback is not taken (the solve would then fail or move).
#[test]
fn a_declined_subject_takes_the_fd_gradient_and_is_recorded() {
    let p = prep(F2, F2_DATA, Some(3), |t| t);
    let (layout, plain) = solve_at(&p);
    let (_, forced) = with_declined(1, || solve_at(&p));
    assert_eq!(forced.fd_gradient_subjects, vec![1]);
    assert!(plain.fd_gradient_subjects.is_empty());
    let worst = layout
        .coords
        .iter()
        .zip(&forced.se_scale)
        .map(|(&k, se)| (forced.params.theta[k] - plain.params.theta[k]).abs() / se)
        .fold(0.0_f64, |a, b| {
            assert!(b.is_finite());
            a.max(b)
        });
    eprintln!("declined vs adjoint β̂: {worst:.3e} SE");
    assert!(worst < DECLINED_BETA, "{worst:.3e}");
}

/// Both solves stop with their gradient within `PROFILE_TOL` (1e-5) SEs of zero, so their
/// β̂ lie within ~2e-5 SE of each other. Realised 1.1e-7 SE (Linux aarch64, macOS arm64).
const DECLINED_BETA: f64 = 1e-5;

/// The fit-level warning, in full: its code, one subject declined, named, with the
/// consequence. Asserted by equality, so dropping any clause of it fails. The same fit
/// without the switch is the straddle: no warning.
#[test]
fn a_profiled_fit_reports_its_fd_profile_gradients() {
    let (p, forced) = with_declined(0, || short_profiled_fit(|t| t));
    let n = p.population.subjects.len();
    let id = &p.population.subjects[0].id;
    let want = format!(
        "W_PROFILE_FD_GRADIENT: 1 of {n} subjects could not be given the exact analytic \
         gradient of the profile solve during this fit (e.g. subject {id}), and their \
         profiled levels' gradients used reconverged finite differences, which are correct \
         but slower."
    );
    let ours = |r: &OuterResult| -> Vec<String> {
        r.warnings
            .iter()
            .filter(|w| w.contains("W_PROFILE_FD_GRADIENT"))
            .cloned()
            .collect()
    };
    assert_eq!(ours(&forced), vec![want], "{:?}", forced.warnings);
    let (_, plain) = short_profiled_fit(|t| t);
    assert!(ours(&plain).is_empty(), "{:?}", plain.warnings);
}

/// The covariance step reports its own: subject 0 declined in the step's solves. Dies when
/// the step drops its stencil's fallbacks, the one place it collects them.
#[test]
fn the_profiled_covariance_step_reports_its_fd_profile_gradients() {
    let p = prep(F2, F2_DATA, Some(2), |t| t);
    let layout = layout_of(&p);
    let o = options(&p);
    let c = ctx(&p, &layout, &o);
    let (_, s) = solve_at(&p);
    let hold = hold_template(&layout, &p.init_params);
    let PackedStart {
        packed: x, fixed, ..
    } = pack_with_bounds(&hold);
    let free: Vec<usize> = (0..x.len()).filter(|&i| !fixed[i]).collect();
    let out = with_declined(0, || {
        c.covariance_at_estimate(&hold, &x, &free, &s.params, &s.etas, &s.kappas)
            .0
    });
    let id = &p.population.subjects[0].id;
    let mine: Vec<&String> = out
        .warnings
        .iter()
        .filter(|w| w.contains("during the covariance step"))
        .collect();
    assert_eq!(mine.len(), 1, "{:?}", out.warnings);
    assert_eq!(
        *mine[0],
        format!(
            "W_PROFILE_FD_GRADIENT: {} of {} subjects could not be given the exact analytic \
             gradient of the profile solve during the covariance step (e.g. subject {id}), and \
             their profiled levels' gradients used reconverged finite differences, which are \
             correct but slower.",
            1,
            p.population.subjects.len()
        )
    );
}

/// #2405: the warning is classified by its code, not its prose. Both phases' text is
/// `GradientFallback` at `Info`; the same text without the code is not, so the straddle
/// is asserted and the arm cannot quietly become a prose match. Dies when the arm is
/// dropped, or moved below a prose arm that claims the text.
#[test]
fn the_profile_fd_gradient_warning_is_classified_by_its_code() {
    use crate::types::{classify_warning, WarningCode, WarningSeverity};
    let p = prep(F2, F2_DATA, Some(2), |t| t);
    let subjects = std::collections::BTreeSet::from([0usize]);
    for during in ["during this fit", "during the covariance step"] {
        let w = profile_fd_gradient_warning(&p.population, &subjects, during).unwrap();
        let c = classify_warning(&w);
        assert_eq!(c.severity, WarningSeverity::Info, "{w}");
        assert_eq!(c.category, WarningCode::GradientFallback, "{w}");
        let bare = w.strip_prefix("W_PROFILE_FD_GRADIENT: ").unwrap();
        let b = classify_warning(bare);
        assert!(
            (b.severity, b.category) != (WarningSeverity::Info, WarningCode::GradientFallback),
            "the prose alone classifies as the code does, so the straddle is gone: {bare}"
        );
    }
}

/// #2405: the work a solve counts, against identities that hold for any run. Every Newton
/// pass takes one level gradient per subject, so a group's gradients are its subjects
/// times its passes; a declined subject's gradient is differenced, two re-solves per
/// level it reads. Dies when a gradient is not counted, a decline is counted as adjoint,
/// or the difference re-solves are dropped.
#[test]
fn a_profile_solve_counts_its_work() {
    let p = prep(F2, F2_DATA, Some(3), |t| t);
    let layout = layout_of(&p);
    let o = options(&p);
    let c = ctx(&p, &layout, &o);
    let warm = ProfileWarm::from_template(&layout, &p.init_params);
    let run = |declined: Option<usize>| {
        let mut work = ProfileCounts::default();
        FORCE_PROFILE_GRADIENT_DECLINE.with(|x| x.set(declined));
        let s = c.solve_counted(&p.init_params, &warm, &mut work);
        FORCE_PROFILE_GRADIENT_DECLINE.with(|x| x.set(None));
        (s.expect("the profile solve failed"), work)
    };
    let (plain_s, plain) = run(None);
    let (forced_s, forced) = run(Some(1));
    // The identities below need both arms to stop after the same steps, so their final
    // criteria must sit clear of `PROFILE_TOL` against the ~1e-7 SE a decline moves the
    // gradient by (#2405 review S3). Measured on Linux aarch64 and macOS arm64 alike: both
    // arms stop at 5.239e-6 (margin 4.761e-6, ~48× the perturbation) after 8 steps; plain
    // 8 re-solves and 11 adjoint gradients, forced 7 adjoint + 4 differenced and
    // 8 + 2·3·4 = 32 re-solves.
    eprintln!(
        "plain {plain:?}, criterion {:.3e} (margin {:.3e}); forced {forced:?}, criterion \
         {:.3e} (margin {:.3e})",
        plain_s.criterion,
        PROFILE_TOL - plain_s.criterion,
        forced_s.criterion,
        PROFILE_TOL - forced_s.criterion
    );
    assert_eq!(plain.fd_gradients, 0, "{plain:?}");
    assert_eq!(plain.adjoint_gradients, plain_s.n_adjoint_gradients);
    assert!(plain.newton_steps >= plain_s.iterations.max(1), "{plain:?}");
    // The straddle: the decline moves gradients from adjoint to differenced and nothing else,
    // since the two solves take the same steps (β̂ within 1e-5 SE, above).
    assert_eq!(
        forced.newton_steps, plain.newton_steps,
        "{forced:?} vs {plain:?}"
    );
    assert!(forced.fd_gradients >= 1, "{forced:?}");
    assert_eq!(
        forced.adjoint_gradients + forced.fd_gradients,
        plain.adjoint_gradients,
        "{forced:?} vs {plain:?}"
    );
    let k1 = layout.reads[1].len();
    assert_eq!(
        forced.resolves,
        plain.resolves + 2 * k1 * forced.fd_gradients,
        "{forced:?} vs {plain:?}"
    );
}

/// #2405: a failed group's work is counted, not dropped with its `Err`. Every group is
/// failed after its first Newton step: the solve fails, and the work it did is there.
/// Dies when the counts are summed after the failure returns.
#[test]
fn a_failed_profile_solve_still_counts_its_work() {
    let p = prep(F2, F2_DATA, Some(3), |t| t);
    let layout = layout_of(&p);
    let o = options(&p);
    let c = ctx(&p, &layout, &o);
    let warm = ProfileWarm::from_template(&layout, &p.init_params);
    let mut work = ProfileCounts::default();
    FORCE_PROFILE_GROUP_FAILURE.with(|x| x.set(Some(1)));
    let s = c.solve_counted(&p.init_params, &warm, &mut work);
    FORCE_PROFILE_GROUP_FAILURE.with(|x| x.set(None));
    eprintln!("failed {work:?}, groups {}", layout.groups.len());
    // Measured on Linux aarch64 and macOS arm64 alike: 3 groups, 3 steps, 3 re-solves, 6
    // adjoint gradients.
    assert!(
        matches!(s, Err(ProfileFailure::NotConverged { .. })),
        "{s:?}"
    );
    let n = p.population.subjects.len();
    assert_eq!(
        work.newton_steps,
        layout.groups.len(),
        "one step per group: {work:?}"
    );
    assert_eq!(
        work.adjoint_gradients,
        2 * n,
        "two passes per subject: {work:?}"
    );
    assert!(
        work.resolves >= n,
        "one line-search trial per subject at least: {work:?}"
    );
}

/// #2405: the `verbose` summary, in full. Each figure lands in its own slot.
#[test]
fn the_profile_work_summary_names_each_count() {
    let c = ProfileCounts {
        newton_steps: 11,
        resolves: 222,
        adjoint_gradients: 33,
        fd_gradients: 4,
    };
    assert_eq!(
        profile_work_summary(7, 2, &c),
        "Profile solves of the outer optimizer: 7 evaluations (2 solve failures, their work \
         counted), 11 Newton steps summed over groups, 222 EBE re-solves, level gradients 33 \
         adjoint / 4 finite-difference"
    );
}

/// #2405: a `verbose` profiled fit sums its solves' work, a failed solve's included. With
/// one subject declined the fit records differenced gradients; with every group failed at
/// its first pass, every solve fails and the fit still records each one's gradients (and
/// takes `failed_result`). Runs `verbose`, so the summary line is
/// executed. Dies when the fit stops accumulating, or accumulates only successful solves.
#[test]
fn a_verbose_profiled_fit_sums_its_profile_work() {
    let verbose_fit = || {
        let p = prep(F2, F2_DATA, Some(2), |t| t);
        let o = FitOptions {
            outer_maxiter: 3,
            run_covariance_step: false,
            verbose: true,
            ..options(&p)
        };
        LAST_PROFILE_WORK.with(|c| c.set(None));
        LAST_PROFILE_SUMMARY.with(|c| *c.borrow_mut() = None);
        let _ = optimize_profiled(&p.parsed.model, &p.population, &p.init_params, &o);
        let (n, failed, work) = LAST_PROFILE_WORK
            .with(std::cell::Cell::get)
            .expect("the fit recorded its work");
        // What was printed is the line of what was summed (#2405 review N2).
        let printed = LAST_PROFILE_SUMMARY.with(|c| c.borrow_mut().take());
        assert_eq!(printed, Some(profile_work_summary(n, failed, &work)));
        (n, failed, work)
    };
    let (n, failed, work) = with_declined(0, verbose_fit);
    eprintln!("declined: {n} evals, {failed} failed, {work:?}");
    assert!(n >= 1);
    assert_eq!(failed, 0, "{work:?}");
    assert!(
        work.fd_gradients >= 1 && work.adjoint_gradients >= 1,
        "{work:?}"
    );

    FORCE_PROFILE_GROUP_FAILURE.with(|x| x.set(Some(0)));
    let out = std::panic::catch_unwind(verbose_fit);
    FORCE_PROFILE_GROUP_FAILURE.with(|x| x.set(None));
    let (n, failed, work) = out.unwrap_or_else(|e| std::panic::resume_unwind(e));
    eprintln!("failing: {n} evals, {failed} failed, {work:?}");
    assert!(n >= 1);
    assert_eq!(failed, n, "every solve fails");
    assert_eq!(work.newton_steps, 0, "{work:?}");
    assert!(work.adjoint_gradients >= n, "{work:?}");
}

/// A level that would log-pack declines the adjoint level gradient (#2401 review row 1):
/// the packed gradient is `∂/∂x`, and reading it as `∂/∂θ` would be off by a factor of θ.
/// One template gives both sides: a lower bound of 0 on the first level makes it log-pack,
/// so a subject reading it declines while a subject reading only other levels is still
/// served. Dies when the check is dropped (the reader is served) or widened to every
/// subject (the other one declines).
#[test]
fn a_level_that_log_packs_declines_the_adjoint_gradient() {
    let p = prep(F2, F2_DATA, Some(3), |t| t);
    let layout = layout_of(&p);
    let o = options(&p);
    let c = ctx(&p, &layout, &o);
    let k = layout.coords[0];
    let mut params = p.init_params.clone();
    params.theta[k] = 1.0;
    params.theta_lower[k] = 0.0;
    assert!(
        theta_packs_log(params.theta_lower[k]),
        "fixture: the level log-packs"
    );
    let reads = |i: usize| {
        layout.reads[i]
            .iter()
            .flat_map(|r| &r.moves)
            .any(|&(j, _)| j == k)
    };
    let reader = (0..p.population.subjects.len())
        .find(|&i| reads(i))
        .unwrap();
    let other = (0..p.population.subjects.len())
        .find(|&i| !reads(i) && !layout.reads[i].is_empty())
        .expect("fixture: a subject reading other levels only");
    let x = pack_params(&params);
    let pt = crate::sens::ode_provider::adjoint::PackedPoint::new(&params, &x).unwrap();
    let zero = DVector::zeros(c.model.n_eta);
    let at = |i: usize| c.total_gradient_adjoint(i, &pt, &c.resolve(i, &params, &zero));
    assert!(
        at(reader).is_none(),
        "the log-packed level's reader must decline"
    );
    assert!(
        at(other).is_some(),
        "a subject not reading it is still served"
    );
}

/// A subject whose inner solve stops short of its mode takes the re-solved differences for
/// its level gradient, and is recorded (#2401 review row 2): the adjoint's identity is exact
/// only at a mode. Measured at the template's levels, against the converged re-solved
/// gradient, in conditional SEs: from a state one inner solve short of its mode, the adjoint
/// was 1.8e-4 off on F2 (`inner_maxiter = 2`) and 4.6 off on `nested_levels_3`
/// (`inner_maxiter = 3`), where the re-solved differences from the same states were 5.8e-7
/// and 2.4e-5. The solve at F2's own `inner_maxiter` records no subject: the straddle.
/// Dies when an off-mode subject still takes the adjoint (nothing recorded). The two solves'
/// levels are not compared: an unconverged inner solve is a different objective.
#[test]
fn a_subject_off_its_mode_takes_the_resolved_level_gradient() {
    let p = prep(F2, F2_DATA, Some(3), |t| t);
    let layout = layout_of(&p);
    let o = options(&p);
    let short = FitOptions {
        inner_maxiter: 2,
        ..o.clone()
    };
    let warm = ProfileWarm::from_template(&layout, &p.init_params);
    let at = |opts: &FitOptions| {
        ctx(&p, &layout, opts)
            .solve(&p.init_params, &warm)
            .unwrap_or_else(|e| panic!("the solve failed: {e:?}"))
    };
    let (full, cut) = (at(&o), at(&short));
    assert!(
        full.fd_gradient_subjects.is_empty(),
        "{:?}",
        full.fd_gradient_subjects
    );
    eprintln!(
        "off-mode: {} of {} subjects on FD, {} adjoint level gradients",
        cut.fd_gradient_subjects.len(),
        p.population.subjects.len(),
        cut.n_adjoint_gradients
    );
    assert!(full.n_adjoint_gradients > 0);
    // At `inner_maxiter = 2` no state of the solve is at its mode, the population inner
    // loop's start states nor the line search's re-solves, so every level gradient must
    // be the re-solved one: each source of the flag is pinned on its own.
    assert_eq!(
        cut.fd_gradient_subjects.len(),
        p.population.subjects.len(),
        "off-mode subjects took the adjoint gradient"
    );
    assert_eq!(
        cut.n_adjoint_gradients, 0,
        "an off-mode state took the adjoint gradient"
    );
}

// ── #2112: bounded profiled blocks ─────────────────────────────────────────────────

/// Rewrites a fixture's `(0.0) profile` block line to declare the box `bx`.
fn boxed(bx: &'static str) -> impl Fn(String) -> String {
    move |t: String| {
        assert!(t.contains("(0.0) profile"), "fixture moved");
        t.replace("(0.0) profile", &format!("{bx} profile"))
    }
}

/// The box-constrained minimiser of `½ΔᵀMΔ + gᵀΔ` subject to `lo ≤ β + Δ ≤ hi`, by
/// enumerating every assignment of each coordinate to {free, lower, upper} and keeping the
/// one that is feasible and satisfies the KKT signs. Shares nothing with `box_qp_step`.
fn kkt_by_enumeration(
    m: &DMatrix<f64>,
    g: &[f64],
    beta: &[f64],
    lo: &[f64],
    hi: &[f64],
) -> Vec<f64> {
    let n = g.len();
    let mut found = Vec::new();
    for code in 0..3usize.pow(n as u32) {
        let s: Vec<usize> = (0..n).map(|k| code / 3usize.pow(k as u32) % 3).collect();
        let mut d = vec![0.0; n];
        for k in 0..n {
            match s[k] {
                1 => d[k] = lo[k] - beta[k],
                2 => d[k] = hi[k] - beta[k],
                _ => {}
            }
        }
        let free: Vec<usize> = (0..n).filter(|&k| s[k] == 0).collect();
        if !free.is_empty() {
            let mff = DMatrix::from_fn(free.len(), free.len(), |a, b| m[(free[a], free[b])]);
            let rhs = DVector::from_iterator(
                free.len(),
                free.iter().map(|&a| {
                    -(g[a]
                        + (0..n)
                            .filter(|&k| s[k] != 0)
                            .map(|k| m[(a, k)] * d[k])
                            .sum::<f64>())
                }),
            );
            let x = mff.cholesky().unwrap().solve(&rhs);
            for (j, &a) in free.iter().enumerate() {
                d[a] = x[j];
            }
        }
        let feasible =
            (0..n).all(|k| beta[k] + d[k] >= lo[k] - 1e-12 && beta[k] + d[k] <= hi[k] + 1e-12);
        let r: Vec<f64> = (0..n)
            .map(|a| g[a] + (0..n).map(|k| m[(a, k)] * d[k]).sum::<f64>())
            .collect();
        let kkt = (0..n).all(|k| match s[k] {
            1 => r[k] >= 0.0,
            2 => r[k] <= 0.0,
            _ => true,
        });
        if feasible && kkt {
            found.push(d);
        }
    }
    assert_eq!(found.len(), 1, "a strictly convex QP has one KKT point");
    found.pop().unwrap()
}

/// A coupled SPD 3×3 Newton matrix, for the closed-form QP tests.
fn q_matrix() -> DMatrix<f64> {
    DMatrix::from_row_slice(3, 3, &[4.0, 1.5, 0.5, 1.5, 3.0, 1.0, 0.5, 1.0, 2.0])
}

/// The bounded step of a borderless 3-θ group (the θ form) from `beta`.
fn q_step(g: &[f64], beta: &[f64], lo: &[f64], hi: &[f64]) -> (Vec<f64>, Vec<Option<f64>>) {
    let grp = bordered_fixture(vec![None; 3], 0);
    let m = q_matrix();
    let blocks = vec![vec![0, 1, 2]];
    box_qp_step(&grp, &m, g, &blocks, beta, lo, hi, &[1.0; 3]).expect("the QP settles")
}

/// Q1: on a coupled 3×3 `M` the bounded step is the KKT point enumerated by hand over
/// every working set, with one coordinate on a bound and the others inside, and it is
/// **not** the clamped Newton step (the straddle, asserted: the clamp lands > 0.1 away).
/// Dies when the QP is replaced by clamping the unconstrained step, or when a θ is never
/// released from the working set.
#[test]
fn the_box_qp_is_the_kkt_point_not_the_clamped_newton_step() {
    let m = q_matrix();
    let (lo, hi, beta) = ([-1.0; 3], [1.0, 1.0, 3.0], [0.0; 3]);
    let g = [-6.0, 1.0, -3.0];
    let want = kkt_by_enumeration(&m, &g, &beta, &lo, &hi);
    let (got, pinned) = q_step(&g, &beta, &lo, &hi);
    let at = |k: usize| want[k] == lo[k] || want[k] == hi[k];
    let on_bound = (0..3).filter(|&k| at(k)).count();
    assert!(
        on_bound >= 1 && on_bound < 3,
        "the fixture straddles: some θ on a bound, some inside ({want:?})"
    );
    for k in 0..3 {
        assert!(got[k].is_finite());
        assert!(
            (got[k] - want[k]).abs() < 1e-12,
            "θ {k}: {got:?} vs {want:?}"
        );
        assert_eq!(pinned[k].is_some(), at(k), "θ {k}: {pinned:?}");
    }
    let newton = -m
        .clone()
        .cholesky()
        .unwrap()
        .solve(&DVector::from_row_slice(&g));
    let clamp_gap = (0..3)
        .map(|k| (newton[k].clamp(lo[k], hi[k]) - want[k]).abs())
        .fold(0.0_f64, f64::max);
    eprintln!(
        "Q1: kkt {want:?}, newton {:?}, clamp gap {clamp_gap:.3}",
        newton.as_slice()
    );
    assert!(
        clamp_gap > 0.1,
        "clamping must miss the KKT point: {clamp_gap}"
    );
}

/// Q1, the release arm: a box on which the active-set path blocks θ 2 at a bound on its way
/// (θ 0 and θ 1 still free then), and the subspace minimum, with θ 0 and θ 1 on their own
/// bounds, has θ 2's multiplier pointing into the box — so θ 2 must leave the working set
/// and ends strictly inside (found by a search over boxes; the enumerated KKT point is the
/// judge). Dies when a θ is never released.
///
/// The second box (#2402 review #2) is one where a held θ's multiplier has a different
/// sign at the start than at the subspace minimum: read at the start gradient, the release
/// rule cycles to the pass cap (`BoxQp`) instead of reaching the KKT point. Each box is
/// asserted against the enumerated KKT point, its pinned set read off that point.
#[test]
fn a_theta_blocked_on_the_way_is_released_at_the_subspace_minimum() {
    let m = q_matrix();
    let (lo, hi, beta) = ([-2.0, -2.0, -0.5], [2.0, 1.0, 0.5], [0.0; 3]);
    let g = [-5.8, 6.1, 0.3];
    let want = kkt_by_enumeration(&m, &g, &beta, &lo, &hi);
    let (got, pinned) = q_step(&g, &beta, &lo, &hi);
    assert!(
        want[2] > lo[2] + 0.05 && want[2] < hi[2] - 0.05,
        "θ 2 ends inside: {want:?}"
    );
    assert_eq!(pinned, [Some(2.0), Some(-2.0), None]);
    for k in 0..3 {
        assert!(
            (got[k] - want[k]).abs() < 1e-12,
            "θ {k}: {got:?} vs {want:?}"
        );
    }

    let (lo, hi) = ([-0.5, -1.25, -1.65], [0.14, 0.13, 0.38]);
    let g = [-6.26, 6.4, 0.16];
    let want = kkt_by_enumeration(&m, &g, &beta, &lo, &hi);
    let (got, pinned) = q_step(&g, &beta, &lo, &hi);
    for k in 0..3 {
        let on = want[k] == lo[k] || want[k] == hi[k];
        assert_eq!(pinned[k].is_some(), on, "θ {k}: {pinned:?} vs {want:?}");
        assert!(
            (got[k] - want[k]).abs() < 1e-12,
            "θ {k}: {got:?} vs {want:?}"
        );
    }
}

/// The line search's trial levels: a θ the full step ends on a bound is the bound itself,
/// bitwise, also where `β + (u − β)` rounds below it (−0.78 → 1.81 is 1.8099999999999998);
/// at a shorter step, and for the free θ, the plain step clamped into the box.
#[test]
fn a_full_step_onto_a_bound_lands_on_it_exactly() {
    let (b, u) = (-0.78, 1.81);
    assert!(b + (u - b) < u, "the pair rounds below the bound");
    let beta = [b, 0.0];
    let step = [u - b, 0.25];
    let pinned = [Some(u), None];
    let (lo, hi) = ([-10.0, -10.0], [u, 10.0]);
    let full = trial_levels(&beta, &step, &pinned, 1.0, &lo, &hi);
    assert_eq!(full[0].to_bits(), u.to_bits());
    assert_eq!(full[1], 0.25);
    let half = trial_levels(&beta, &step, &pinned, 0.5, &lo, &hi);
    assert_eq!(half[0], b + 0.5 * (u - b));
    assert!(half[0] < u);
}

/// Q3: a θ that **starts** on its lower bound stays there when its gradient points out of
/// the box, and is released when it points in — both in one test, so a flipped multiplier
/// test or a release of every held θ each pass reddens one half.
#[test]
fn a_theta_on_a_bound_is_held_or_released_by_its_multiplier() {
    let m = q_matrix();
    let (lo, hi, beta) = ([-1.0; 3], [1.0; 3], [-1.0, 0.0, 0.0]);
    let mut kept = 0;
    for (g, held) in [([1.0, -0.5, 0.2], true), ([-3.0, -0.5, 0.2], false)] {
        let want = kkt_by_enumeration(&m, &g, &beta, &lo, &hi);
        let (got, pinned) = q_step(&g, &beta, &lo, &hi);
        assert_eq!(
            want[0] == 0.0,
            held,
            "the oracle agrees with the case: {want:?}"
        );
        assert_eq!(pinned[0], held.then_some(-1.0), "g = {g:?}: {pinned:?}");
        for k in 0..3 {
            assert!(
                (got[k] - want[k]).abs() < 1e-12,
                "g = {g:?}, θ {k}: {got:?} vs {want:?}"
            );
        }
        kept += usize::from(held);
    }
    assert_eq!(kept, 1, "one case keeps the θ, one releases it");
}

/// Q2: under a border, the bounded step with one θ held equals the bounded step on the θ
/// form (`TᵀMT`, `Tᵀg`, no borders) of the same group. Dies when the bordered path ignores
/// the held mask.
#[test]
fn the_bordered_box_step_is_the_theta_form_box_step() {
    let border_of = vec![None, Some(0), Some(0), Some(0)];
    let blocks = vec![vec![0, 1], vec![2, 4], vec![3]];
    let g = bordered_fixture(border_of, 1);
    let n = g.n_levels();
    let m = block_spd(n, &blocks, 11);
    let gy: Vec<f64> = (0..n).map(|l| (l as f64 * 0.7).sin() + 0.3).collect();
    let beta = [0.0; 4];
    let theta_m = g.theta_matrix(&m);
    let theta_g = g.theta_gradient(&gy);
    let unbounded = -theta_m
        .clone()
        .cholesky()
        .unwrap()
        .solve(&DVector::from_vec(theta_g.clone()));
    // A box that cuts the θ the unbounded step moves most.
    let k = unbounded.iamax();
    let cut = 0.5 * unbounded[k];
    let (mut lo, mut hi) = ([f64::NEG_INFINITY; 4], [f64::INFINITY; 4]);
    if cut > 0.0 {
        hi[k] = cut;
    } else {
        lo[k] = cut;
    }
    let (bordered, pinned) = box_qp_step(&g, &m, &gy, &blocks, &beta, &lo, &hi, &[1.0; 4]).unwrap();
    let theta_form = bordered_fixture(vec![None; 4], 0);
    let (dense, _) = box_qp_step(
        &theta_form,
        &theta_m,
        &theta_g,
        &[vec![0, 1, 2, 3]],
        &beta,
        &lo,
        &hi,
        &[1.0; 4],
    )
    .unwrap();
    assert_eq!(
        pinned[k],
        Some(cut),
        "the cut θ ends on its bound: {pinned:?}"
    );
    for a in 0..4 {
        assert!(bordered[a].is_finite());
        assert!(
            (bordered[a] - dense[a]).abs() < 1e-12 * (1.0 + dense[a].abs()),
            "θ {a}: {bordered:?} vs {dense:?}"
        );
    }
    assert!(
        (0..4)
            .filter(|&a| a != k)
            .any(|a| (dense[a] - unbounded[a]).abs() > 1e-3),
        "holding the cut θ moves the others"
    );
}

/// The solve of `edit`'s fixture at the template's structural point.
fn solve_boxed(
    model: &str,
    data: &str,
    studies: Option<usize>,
    edit: impl Fn(String) -> String,
) -> (crate::api::PreparedRun, ProfileLayout, ProfileSolve) {
    let p = prep(model, data, studies, edit);
    let layout = layout_of(&p);
    let o = options(&p);
    let s = ctx(&p, &layout, &o)
        .solve(
            &p.init_params,
            &ProfileWarm::from_template(&layout, &p.init_params),
        )
        .expect("the solve converges");
    (p, layout, s)
}

/// The θ index of the level `name` of `p`'s block.
fn theta_named(p: &crate::api::PreparedRun, name: &str) -> usize {
    p.init_params
        .theta_names
        .iter()
        .position(|n| n == name)
        .unwrap_or_else(|| panic!("{name}"))
}

/// N1: on F1 with `(0.0, -10.0, 3.0)`, the level the unbounded solve puts at 4.39 is held
/// at **exactly** 3.0, bitwise, while one the unbounded solve puts at 2.62 — 0.38 inside the
/// same bound — ends strictly inside it with its gradient at the tolerance (the straddle,
/// asserted). F1 is linear in (η, κ), so the bounded solve still takes one Newton step.
/// Dies when the level is written as `β + Δ` instead of the bound, or when activity is
/// measured with a tolerance.
#[test]
fn a_bounded_profile_holds_a_level_exactly_on_its_bound() {
    let (p, layout, s) = solve_boxed(F1, F1_DATA, None, boxed("(0.0, -10.0, 3.0)"));
    let pinned = theta_named(&p, "PLACEBO[STUDY=3,TIME=0]");
    let free = theta_named(&p, "PLACEBO[STUDY=5,TIME=0]");
    assert_eq!(s.params.theta[pinned].to_bits(), 3.0_f64.to_bits());
    let v = s.params.theta[free];
    assert!(v.is_finite() && v < 3.0 && v > 2.0, "{v}");
    assert!(layout
        .at_bound(&s.params.theta)
        .contains(&(pinned, 3.0, true)));
    assert!(!layout.at_bound(&s.params.theta).iter().any(|h| h.0 == free));
    assert!(
        s.criterion.is_finite() && s.criterion <= PROFILE_TOL,
        "{}",
        s.criterion
    );
    assert_eq!(s.iterations, 1);
    let unbounded = solve_boxed(F1, F1_DATA, None, |t| t).2;
    assert!(
        unbounded.params.theta[pinned] > 3.0 && unbounded.params.theta[free] < 3.0,
        "the box binds one level and not the other"
    );
}

/// N1 on the **lower** side (#2402 review #1): on F1 with `(0.0, -1.5, 10.0)`, the level the unbounded solve puts at −2.03 is held at exactly −1.5,
/// reported on its lower side, and one the unbounded solve puts at −1.33 — 0.17 inside the
/// same bound — ends strictly inside it, with the projected criterion met (the straddle,
/// asserted). Dies when the projected criterion reads `|g|·se` at a lower bound (the solve
/// can then never stop) or when `at_bound` drops its lower arm.
#[test]
fn a_bounded_profile_holds_a_level_exactly_on_its_lower_bound() {
    let (p, layout, s) = solve_boxed(F1, F1_DATA, None, boxed("(0.0, -1.5, 10.0)"));
    let pinned = theta_named(&p, "PLACEBO[STUDY=3,TIME=4]");
    let free = theta_named(&p, "PLACEBO[STUDY=5,TIME=4]");
    assert_eq!(s.params.theta[pinned].to_bits(), (-1.5_f64).to_bits());
    assert_eq!(
        layout.at_bound(&s.params.theta),
        vec![(pinned, -1.5, false)]
    );
    let v = s.params.theta[free];
    assert!(v.is_finite() && v > -1.5 && v < -1.0, "{v}");
    assert!(
        s.criterion.is_finite() && s.criterion <= PROFILE_TOL,
        "{}",
        s.criterion
    );
    assert_eq!(s.iterations, 1);
    let unbounded = solve_boxed(F1, F1_DATA, None, |t| t).2;
    assert!(
        unbounded.params.theta[pinned] < -1.5 && unbounded.params.theta[free] > -1.5,
        "the box binds one level and not the other"
    );
}

/// P4 (#2112): a profiled level with `lower = 0` reaches **exactly** 0, where an
/// unprofiled one would only approach it through the log packing's `1e-10` floor. F1 with
/// `contrast = none` (its `ETA_E0` dropped and `TVE0` fixed, which the binder requires for an
/// unconstrained block) and `(1.0, 0.0, 10.0)`: four levels end on 0.0, the others inside.
#[test]
fn a_profiled_level_with_a_zero_lower_bound_reaches_exactly_zero() {
    let (p, layout, s) = solve_boxed(F1, F1_DATA, None, |t| {
        boxed("(1.0, 0.0, 10.0)")(
            t.replace("contrast = sum_to_zero_within", "contrast = none")
                .replace("  omega ETA_E0 ~ 4.0 (sd)\n", "")
                .replace(" + ETA_E0", "")
                .replace("theta TVE0(45.0, 0.0, 200.0)", "theta TVE0(45.0, FIX)"),
        )
    });
    for &k in &layout.coords {
        assert!(!crate::estimation::parameterization::theta_packs_log(
            p.init_params.theta_lower[k]
        ));
        assert!(s.params.theta[k] >= 0.0, "{}", p.init_params.theta_names[k]);
    }
    let hits = layout.at_bound(&s.params.theta);
    assert_eq!(hits.len(), 4, "{hits:?}");
    for &(k, v, upper) in &hits {
        assert!(!upper);
        assert_eq!(v.to_bits(), 0.0_f64.to_bits());
        assert_eq!(s.params.theta[k].to_bits(), 0.0_f64.to_bits());
    }
    assert!(s.criterion <= PROFILE_TOL, "{}", s.criterion);
}

/// A warm start outside the box is clamped into it before the first inner solve. The start
/// here is the **unbounded** solve's levels — a fit resumed after a box was added — where
/// the gradient is already at the tolerance, so without the clamp the solve stops at once
/// on levels outside the box (the line search's own clamp never runs).
#[test]
fn a_start_outside_the_box_is_clamped_into_it() {
    let (p, layout, inside) = solve_boxed(F1, F1_DATA, None, boxed("(0.0, -10.0, 3.0)"));
    let o = options(&p);
    let unbounded = solve_boxed(F1, F1_DATA, None, |t| t).2;
    let outside = ProfileWarm::from_solve(&layout, &unbounded);
    assert!(
        outside.beta.iter().any(|&b| b > 3.0),
        "the start leaves the box"
    );
    let s = ctx(&p, &layout, &o)
        .solve(&p.init_params, &outside)
        .expect("the solve converges");
    for &k in &layout.coords {
        let v = s.params.theta[k];
        assert!(
            v.is_finite() && v <= 3.0,
            "{}: {v}",
            p.init_params.theta_names[k]
        );
    }
    assert_eq!(
        layout.at_bound(&s.params.theta),
        layout.at_bound(&inside.params.theta)
    );
    assert!(
        (s.ofv - inside.ofv).abs() < D0_BOUND,
        "{} vs {}",
        s.ofv,
        inside.ofv
    );
}

/// D0b: the bounded solve's objective is the unprofiled objective at its own joint point
/// `(θ, β̂)`, and equals the unbounded solve with its pinned levels frozen at the bound
/// (the #1891 offset path). F1 (block-diagonal, linear) and F2 cut to three studies
/// (nonlinear, two levels pinned) — a bounded solve that left a different point dies.
/// Realised on F2 (Linux aarch64 and macOS arm64): 3.0e-10 to the unprofiled objective,
/// 8.7e-10 to the frozen solve; F1 identical in every printed digit. D0's band.
#[test]
fn the_bounded_objective_is_the_frozen_unbounded_objective() {
    for (model, data, n, bx) in [
        (F1, F1_DATA, None, "(0.0, -10.0, 3.0)"),
        (F2, F2_DATA, Some(3), "(0.0, -10.0, 1.0)"),
    ] {
        let (p, layout, s) = solve_boxed(model, data, n, boxed(bx));
        let hits = layout.at_bound(&s.params.theta);
        assert!(!hits.is_empty() && hits.len() < layout.coords.len());
        let o = options(&p);
        let mu = compute_mu_k(&p.parsed.model, &s.params.theta, o.mu_referencing);
        let (e, h, _, k) = run_inner_loop_warm_seeded(
            &p.parsed.model,
            &p.population,
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
                &p.parsed.model,
                &p.population,
                &s.params,
                &e,
                &h,
                &k,
                &o,
            );
        // The unbounded solve with the pinned levels frozen at their bound.
        let mut q = prep(model, data, n, |t| t);
        for &(k, v, _) in &hits {
            q.init_params.theta[k] = v;
            q.init_params.theta_fixed[k] = true;
        }
        let frozen_layout = layout_of(&q);
        let qo = options(&q);
        let frozen = ctx(&q, &frozen_layout, &qo)
            .solve(
                &q.init_params,
                &ProfileWarm::from_template(&frozen_layout, &q.init_params),
            )
            .unwrap();
        eprintln!(
            "D0b {model}: bounded {:.12} plain {:.12} frozen {:.12} ({} pinned)",
            s.ofv,
            plain,
            frozen.ofv,
            hits.len()
        );
        assert!(s.ofv.is_finite() && plain.is_finite() && frozen.ofv.is_finite());
        assert!((s.ofv - plain).abs() < D0_BOUND, "{} vs {}", s.ofv, plain);
        assert!(
            (s.ofv - frozen.ofv).abs() < D0_BOUND,
            "{} vs {}",
            s.ofv,
            frozen.ofv
        );
    }
}

/// The box-constrained oracle: projected Newton over every coordinate at once on
/// [`total_ofv`] — the coordinates on a bound whose FD gradient points out of the box held,
/// a dense FD Newton step on the rest, the iterate projected onto the box — until the free
/// step vanishes. It shares only the objective with the solve's active-set QP.
fn oracle_box_min(
    c: &ProfileCtx<'_>,
    params: &ModelParameters,
    start: &[f64],
    lo: &[f64],
    hi: &[f64],
) -> (f64, Vec<f64>) {
    let n = start.len();
    let mut b: Vec<f64> = (0..n).map(|j| start[j].clamp(lo[j], hi[j])).collect();
    let f = |x: &[f64]| total_ofv(c, params, x);
    let at = |b: &[f64], d: &[(usize, f64)]| {
        let mut x = b.to_vec();
        for &(j, v) in d {
            x[j] += v;
        }
        f(&x)
    };
    for _ in 0..60 {
        let hg = 1e-4;
        let g: Vec<f64> = (0..n)
            .map(|j| (at(&b, &[(j, hg)]) - at(&b, &[(j, -hg)])) / (2.0 * hg))
            .collect();
        let free: Vec<usize> = (0..n)
            .filter(|&j| !((b[j] == lo[j] && g[j] > 0.0) || (b[j] == hi[j] && g[j] < 0.0)))
            .collect();
        let hh = 1e-2;
        let f0 = f(&b);
        let nf = free.len();
        let mut h = DMatrix::zeros(nf, nf);
        for (a, &j) in free.iter().enumerate() {
            h[(a, a)] = (at(&b, &[(j, hh)]) - 2.0 * f0 + at(&b, &[(j, -hh)])) / (hh * hh);
            for (c2, &k) in free.iter().enumerate().take(a) {
                let v = (at(&b, &[(j, hh), (k, hh)])
                    - at(&b, &[(j, hh), (k, -hh)])
                    - at(&b, &[(j, -hh), (k, hh)])
                    + at(&b, &[(j, -hh), (k, -hh)]))
                    / (4.0 * hh * hh);
                h[(a, c2)] = v;
                h[(c2, a)] = v;
            }
        }
        let step = h
            .cholesky()
            .expect("oracle Hessian PD")
            .solve(&DVector::from_iterator(nf, free.iter().map(|&j| g[j])));
        for (a, &j) in free.iter().enumerate() {
            b[j] = (b[j] - step[a]).clamp(lo[j], hi[j]);
        }
        if step.amax() < 1e-9 {
            break;
        }
    }
    (f(&b), b)
}

/// D1b: on F2 cut to three studies with `(0.0, -10.0, 1.0)` — nonlinear in η, two levels
/// pinned in two coupled studies and the third study free — the bounded solve's objective
/// and levels are the box-constrained oracle's. Dies when the QP is replaced by clamping
/// the Newton step, which on a coupled study is not the constrained minimum. Realised
/// (Linux aarch64 and macOS arm64): ΔOFV 5.1e-10 and a worst level gap of 1.14e-5 SE, inside
/// D3's bands (196× and 8.8× headroom).
///
/// Run on both level-gradient engines (#2109), each asserted by its counts: the adjoint
/// (F2's default route), where a pinned level's total gradient is its multiplier, not ~0,
/// and the reconverged FD differences (`gradient = fd`).
fn bounded_profile_is_the_box_constrained_minimum(adjoint: bool) {
    let edit = |t: String| {
        let t = boxed("(0.0, -10.0, 1.0)")(t);
        if adjoint {
            t
        } else {
            forced_fd(t)
        }
    };
    let (p, layout, s) = solve_boxed(F2, F2_DATA, Some(3), edit);
    assert_eq!(
        profile_gradient_is_analytic(&p.parsed.model, &options(&p)),
        adjoint
    );
    assert!(
        s.fd_gradient_subjects.is_empty(),
        "{:?}",
        s.fd_gradient_subjects
    );
    if adjoint {
        assert!(
            s.n_adjoint_gradients >= p.population.subjects.len(),
            "{} adjoint gradients for {} subjects",
            s.n_adjoint_gradients,
            p.population.subjects.len()
        );
    } else {
        assert_eq!(s.n_adjoint_gradients, 0);
    }
    let o = options(&p);
    let c = ctx(&p, &layout, &o);
    let warm = ProfileWarm::from_template(&layout, &p.init_params);
    let (o_star, b_star) =
        oracle_box_min(&c, &p.init_params, &warm.beta, &layout.lower, &layout.upper);
    let worst_beta = layout
        .coords
        .iter()
        .zip(&b_star)
        .zip(&s.se_scale)
        .map(|((&k, &b), &se)| ((s.params.theta[k] - b) / se).abs())
        .fold(0.0_f64, |a, v| {
            assert!(v.is_finite());
            a.max(v)
        });
    eprintln!(
        "D1b adjoint={adjoint}: bounded {:.10} oracle {:.10} (Δ {:.2e}), worst level gap \
         {:.2e} SE, {} steps",
        s.ofv,
        o_star,
        s.ofv - o_star,
        worst_beta,
        s.iterations
    );
    assert!(s.ofv.is_finite() && o_star.is_finite());
    assert!(
        layout.at_bound(&s.params.theta).len() == 2,
        "two levels pinned"
    );
    assert!(
        (s.ofv - o_star).abs() < D3_OFV_BOUND,
        "{} vs {}",
        s.ofv,
        o_star
    );
    assert!(worst_beta < D3_BETA_BOUND, "{worst_beta}");
}

#[test]
fn the_bounded_profile_is_the_box_constrained_minimum_on_the_adjoint() {
    bounded_profile_is_the_box_constrained_minimum(true);
}

#[test]
fn the_bounded_profile_is_the_box_constrained_minimum_on_fd() {
    bounded_profile_is_the_box_constrained_minimum(false);
}

/// B1: a box no level reaches, `(0.0, -1e9, 1e9)`, leaves the solve **bit-identical** to
/// the unboxed declaration — β̂, OFV, step count, criterion and level scale — on F1
/// (block-diagonal, the θ form) and under a global `sum_to_zero` (the bordered step). So
/// the bounded loop with an empty working set is today's step, code path for code path.
#[test]
fn an_unreached_box_is_bit_identical_to_no_box() {
    let wide = boxed("(0.0, -1e9, 1e9)");
    let cases: Vec<(Box<dyn Fn() -> crate::api::PreparedRun>, &str)> = vec![
        (Box::new(|| prep(F1, F1_DATA, None, |t| t)), "F1"),
        (
            Box::new(|| prep(F1, F1_DATA, None, boxed("(0.0, -1e9, 1e9)"))),
            "F1 boxed",
        ),
        (Box::new(|| global_prep(2)), "G"),
        (
            Box::new(move || prep(F2, F2_DATA, Some(2), |t| wide(t.replace(WITHIN, GLOBAL)))),
            "G boxed",
        ),
    ];
    let solved: Vec<(ProfileLayout, ProfileSolve)> = cases
        .iter()
        .map(|(mk, _)| {
            let p = mk();
            let layout = layout_of(&p);
            let o = options(&p);
            let s = ctx(&p, &layout, &o)
                .solve(
                    &p.init_params,
                    &ProfileWarm::from_template(&layout, &p.init_params),
                )
                .unwrap();
            (layout, s)
        })
        .collect();
    for pair in solved.chunks(2) {
        let ((la, a), (lb, b)) = (&pair[0], &pair[1]);
        assert!(la.lower.iter().all(|v| v.is_infinite()));
        assert!(lb.lower.iter().all(|&v| v == -1e9));
        assert_eq!(la.borders.len(), lb.borders.len());
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a.params.theta), bits(&b.params.theta));
        assert_eq!(a.ofv.to_bits(), b.ofv.to_bits());
        assert_eq!(a.iterations, b.iterations);
        assert_eq!(a.criterion.to_bits(), b.criterion.to_bits());
        assert_eq!(bits(&a.level_scale), bits(&b.level_scale));
    }
    assert!(solved[2].0.borders.len() == 1, "G takes the bordered step");
}

/// W1 (#2112), the at-bound warning's text over its input space, as a pure function: one
/// level on a lower bound, and several on mixed sides, each without and with the covariance
/// step's sentence on standard errors. Whole-string equality, so deleting any sentence, or
/// a level's side, fails; the SE sentence is asserted on both sides of its gate in this one
/// test. Never "optimizer bound" (the box is the profile solve's, not NLopt's), and
/// `BoundaryEstimate`, so every boundary consumer reads it.
#[test]
fn the_profiled_bound_warning_names_each_level_and_its_side() {
    let names: Vec<String> = ["TVE0", "PLACEBO[STUDY=1]", "PLACEBO[STUDY=2]"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let first = "Profiled level(s) on a declared bound: PLACEBO[STUDY=1] (-0.5000 at lower \
                 bound). The profile solve holds each at its bound. This often indicates \
                 non-identifiability or a too-tight bound; inspect the affected level(s) and \
                 consider relaxing the bound.";
    let se = " Their standard errors are not available; the other estimates' standard errors \
              are computed with them held at the bound.";
    let (one, entry) =
        crate::api::profiled_bound_warning(&names, &[(1, -0.5, false)], false).expect("one hit");
    assert_eq!(one, first);
    assert_eq!(entry.category, crate::types::WarningCode::BoundaryEstimate);
    let (one_se, _) =
        crate::api::profiled_bound_warning(&names, &[(1, -0.5, false)], true).expect("one hit");
    assert_eq!(one_se, format!("{first}{se}"));
    let (several, several_entry) =
        crate::api::profiled_bound_warning(&names, &[(1, -0.5, false), (2, 2.5, true)], false)
            .unwrap();
    // The structured details (#2402 review #4): each level with its estimate, bound and
    // side, both sides present.
    assert_eq!(
        several_entry.details,
        Some(serde_json::json!({ "parameters": [
            { "parameter": "PLACEBO[STUDY=1]", "estimate": -0.5, "bound": -0.5, "side": "lower" },
            { "parameter": "PLACEBO[STUDY=2]", "estimate": 2.5, "bound": 2.5, "side": "upper" },
        ]}))
    );
    assert!(
        several.starts_with(
            "Profiled level(s) on a declared bound: PLACEBO[STUDY=1] (-0.5000 at lower bound), \
             PLACEBO[STUDY=2] (2.5000 at upper bound). "
        ),
        "{several}"
    );
    for w in [&one, &one_se, &several] {
        assert!(!w.contains("optimizer bound") && !w.contains("TVE0"), "{w}");
    }
    assert!(crate::api::profiled_bound_warning(&names, &[], true).is_none());
}

/// F1b's optimum under `(0.0, -10.0, 3.0)`, as `(TVE0, EMAX, ED50, ET50, ω², κ²)`: a fit
/// to convergence from the fixture's start (OFV 104.717942, `PLACEBO[STUDY=3,TIME=0]` on
/// its upper bound, the #2112 plan's M3), macOS arm64. A covariance step needs an
/// optimum: at the fixture's start `A_p` is not positive definite.
const F1B_UPPER_OPT: [f64; 6] = [
    48.771349936678604,
    12.990192620251463,
    26.78565976686997,
    3.773073521076787,
    4.78392827418847,
    152.43769919945854,
];
/// The same under `(0.0, -1.5, 10.0)` (OFV 98.826099, `PLACEBO[STUDY=3,TIME=4]` on its
/// lower bound).
const F1B_LOWER_OPT: [f64; 6] = [
    48.61340939457224,
    13.26611176798223,
    32.28820509677258,
    3.97672798991109,
    4.834982172888694,
    154.24285500544312,
];

/// F1 with its block line edited by `edit`, evaluated (`outer_maxiter = 0`) at `start`
/// (`(TVE0, EMAX, ED50, ET50, ω², κ²)`, the levels left for the profile solve) or at the
/// fixture's inits, with the covariance step `method` when `cov`.
fn f1b_evaluated(
    edit: &dyn Fn(String) -> String,
    start: Option<&[f64; 6]>,
    cov: bool,
    method: CovarianceMethod,
) -> (crate::api::PreparedRun, crate::types::FitResult) {
    let p = prep(F1, F1_DATA, None, edit);
    let o = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: cov,
        covariance_method: method,
        ..options(&p)
    };
    let f = crate::api::fit(&p.parsed.model, &p.population, &f1b_start(&p, start), &o)
        .unwrap_or_else(|e| panic!("{e}"));
    (p, f)
}

/// `p`'s inits with the structural θ, ω² and κ² of `start` written in.
fn f1b_start(
    p: &crate::api::PreparedRun,
    start: Option<&[f64; 6]>,
) -> crate::types::ModelParameters {
    let mut params = p.init_params.clone();
    if let Some(s) = start {
        params.theta[..4].copy_from_slice(&s[..4]);
        params.omega =
            crate::types::OmegaMatrix::from_diagonal(&[s[4]], params.omega.eta_names.clone());
        let iov = params.omega_iov.as_ref().unwrap().eta_names.clone();
        params.omega_iov = Some(crate::types::OmegaMatrix::from_diagonal(&[s[5]], iov));
    }
    params
}

/// C4 and W1 through `fit()` (#2112), evaluation only on F1, at the optimum of
/// `(0.0, -10.0, 3.0)`, where `PLACEBO[STUDY=3,TIME=0]` ends on its upper bound, and of
/// `(0.0, -1.5, 10.0)`, where `PLACEBO[STUDY=3,TIME=4]` ends on its lower one. The
/// covariance step runs with the pinned level held (`profiled_levels_at_bound`): its row and
/// column of the covariance are exactly zero, every other diagonal is positive, the
/// sum-to-zero dependent of its study keeps a finite SE, and the at-bound warning carries
/// its SE sentence. With the step off, the field is the same and the warning has no SE
/// sentence: both sides of that gate here. The unboxed fit is the other side of
/// `estimate_near_boundary`.
#[test]
fn a_fit_with_a_level_on_its_bound_holds_it_in_the_covariance_step() {
    let run = f1b_evaluated;
    // The covariance holds exactly `held`: zero rows and columns for them, a positive
    // diagonal for every other estimated θ.
    let assert_held = |p: &crate::api::PreparedRun, f: &crate::types::FitResult, held: &[usize]| {
        let c = f
            .covariance_matrix
            .as_ref()
            .unwrap_or_else(|| panic!("no covariance: {:?}", f.warnings));
        assert_eq!(f.profiled_levels_at_bound, held);
        for k in 0..f.theta.len() {
            if held.contains(&k) {
                assert!(
                    (0..c.ncols()).all(|j| c[(k, j)] == 0.0 && c[(j, k)] == 0.0),
                    "{}",
                    p.init_params.theta_names[k]
                );
                assert_eq!(f.se_theta.as_ref().unwrap()[k], 0.0);
            } else if !p.init_params.theta_fixed[k] {
                assert!(
                    c[(k, k)] > 0.0,
                    "{}: {}",
                    p.init_params.theta_names[k],
                    c[(k, k)]
                );
            }
        }
    };
    let bx = boxed("(0.0, -10.0, 3.0)");
    let up = Some(&F1B_UPPER_OPT);
    let (p, with_cov) = run(&bx, up, true, CovarianceMethod::Hessian);
    let pinned = [(theta_named(&p, "PLACEBO[STUDY=3,TIME=0]"), 3.0, true)];
    let held = [pinned[0].0];
    let names = &p.init_params.theta_names;
    let at = |with_se| {
        crate::api::profiled_bound_warning(names, &pinned, with_se)
            .unwrap()
            .0
    };
    assert!(
        with_cov.warnings.contains(&at(true)),
        "{:?}",
        with_cov.warnings
    );
    assert!(!with_cov.warnings.contains(&at(false)));
    assert_held(&p, &with_cov, &held);
    assert_eq!(
        with_cov.covariance_status,
        crate::types::CovarianceStatus::Computed,
        "{:?}",
        with_cov.warnings
    );
    assert!(crate::model_selection::estimate_near_boundary(&with_cov));
    // A pinned level's study keeps its sum-to-zero dependent, whose delta-method SE is
    // over the study's free levels alone.
    let c = with_cov.covariance_matrix.as_ref().unwrap();
    let levels = &crate::api::theta_level_values(&p.parsed.model, &with_cov.theta, Some(c), None)
        .unwrap()["PLACEBO"];
    let dep = levels
        .iter()
        .find(|l| l.theta_index.is_none() && l.label.contains("STUDY=3"))
        .expect("study 3 has a dependent level");
    let dep_se = dep.se.unwrap_or_else(|| panic!("{}: no SE", dep.label));
    assert!(
        dep_se.is_finite() && dep_se > 0.0,
        "{}: {dep_se}",
        dep.label
    );

    let (_, no_cov) = run(&bx, up, false, CovarianceMethod::Hessian);
    assert!(
        no_cov.warnings.contains(&at(false)),
        "{:?}",
        no_cov.warnings
    );
    assert!(!no_cov.warnings.contains(&at(true)));
    assert_eq!(no_cov.profiled_levels_at_bound, held);
    assert!(crate::model_selection::estimate_near_boundary(&no_cov));

    // The lower side (#2402 review #1), and `rsr` (#2108), which runs on the same held
    // layout. `s` is on F2b (`tests/profiled_levels.rs`): F1's 6 subjects leave `S`
    // rank-deficient whatever is held.
    for method in [CovarianceMethod::Hessian, CovarianceMethod::Sandwich] {
        let (lp, lower) = run(
            &boxed("(0.0, -1.5, 10.0)"),
            Some(&F1B_LOWER_OPT),
            true,
            method,
        );
        let k = theta_named(&lp, "PLACEBO[STUDY=3,TIME=4]");
        assert_eq!(lower.theta[k].to_bits(), (-1.5_f64).to_bits());
        assert!(
            lower.warnings.contains(
                &crate::api::profiled_bound_warning(
                    &lp.init_params.theta_names,
                    &[(k, -1.5, false)],
                    true
                )
                .unwrap()
                .0
            ),
            "{method:?}: {:?}",
            lower.warnings
        );
        assert_held(&lp, &lower, &[k]);
        assert_eq!(lower.covariance_method, Some(method));
    }

    // A box that holds every free level (#2416 review row 6): the held layout has no
    // coordinates left, and the step is the structural one around 18 offsets.
    let (tp, tight) = run(
        &boxed("(0.0, -0.01, 0.01)"),
        up,
        true,
        CovarianceMethod::Hessian,
    );
    let levels: Vec<usize> = (0..tight.theta.len())
        .filter(|&k| {
            tp.init_params.theta_names[k].starts_with("PLACEBO[") && !tp.init_params.theta_fixed[k]
        })
        .collect();
    assert_eq!(levels.len(), 18, "every free level");
    assert_held(&tp, &tight, &levels);
    assert_eq!(
        tight.covariance_status,
        crate::types::CovarianceStatus::Computed,
        "{:?}",
        tight.warnings
    );

    let (_, unboxed) = run(&|t| t, None, false, CovarianceMethod::Hessian);
    assert!(!unboxed
        .warnings
        .iter()
        .any(|w| w.contains("declared bound")));
    assert!(unboxed.profiled_levels_at_bound.is_empty());
    assert!(!crate::model_selection::estimate_near_boundary(&unboxed));
}

/// `run_covariance` reports the levels its replayed step held (#2416 review row 2), not the
/// ones the incoming fit listed: the field and the at-bound warning are rebuilt, the old
/// warning dropped, exactly one left. Three incoming fits on F1b's upper optimum:
/// - (a) run without the step: its warning has no SE sentence; the replay's has it;
/// - (b) a fit with neither field nor warning (a bundle saved before the field existed):
///   both are restored, and the console prints the held level's SE as `N/A`, not `0.0`;
/// - (c) (b) with the held level edited to 2.9, inside its box: the replay's re-solve moves
///   it back onto the bound and holds it, so the field comes from the step, not from the
///   edited estimates, which name no level on a bound.
#[test]
fn run_covariance_reports_the_levels_its_step_held() {
    let bx = boxed("(0.0, -10.0, 3.0)");
    let (p, fit) = f1b_evaluated(&bx, Some(&F1B_UPPER_OPT), false, CovarianceMethod::Hessian);
    let k = theta_named(&p, "PLACEBO[STUDY=3,TIME=0]");
    let names = &p.init_params.theta_names;
    let at = |with_se| {
        crate::api::profiled_bound_warning(names, &[(k, 3.0, true)], with_se)
            .unwrap()
            .0
    };
    assert!(fit.covariance_matrix.is_none());
    assert_eq!(fit.profiled_levels_at_bound, [k]);
    assert!(fit.warnings.contains(&at(false)), "{:?}", fit.warnings);
    let o = FitOptions {
        covariance_method: CovarianceMethod::Hessian,
        ..options(&p)
    };
    let replay = |f: &crate::types::FitResult| {
        crate::run_covariance(f, Some(&p.parsed.model), Some(&p.population), &o)
            .unwrap_or_else(|e| panic!("{e}"))
    };
    let reports_held = |out: &crate::types::FitResult, cell: &str| {
        assert!(
            out.covariance_matrix.is_some(),
            "{cell}: {:?}",
            out.warnings
        );
        assert_eq!(out.profiled_levels_at_bound, [k], "{cell}");
        let n = |w: &str| out.warnings.iter().filter(|x| x.as_str() == w).count();
        assert_eq!(
            (n(&at(true)), n(&at(false))),
            (1, 0),
            "{cell}: {:?}",
            out.warnings
        );
        let entries: Vec<_> = out
            .warnings_structured
            .iter()
            .filter(|e| {
                e.message
                    .starts_with("Profiled level(s) on a declared bound:")
            })
            .collect();
        assert_eq!(entries.len(), 1, "{cell}");
        assert_eq!(entries[0].message, at(true), "{cell}");
        assert_eq!(
            entries[0].category,
            crate::types::WarningCode::BoundaryEstimate,
            "{cell}"
        );
        assert!(entries[0].details.is_some(), "{cell}");
    };
    reports_held(&replay(&fit), "(a)");

    let mut stale = fit.clone();
    stale.profiled_levels_at_bound.clear();
    stale.warnings.retain(|w| *w != at(false));
    stale.warnings_structured.retain(|e| e.message != at(false));
    let out = replay(&stale);
    reports_held(&out, "(b)");
    let summary = crate::io::output::format_summary(&out);
    let row = summary
        .lines()
        .find(|l| l.trim_start().starts_with("PLACEBO[STUDY=3,TIME=0]"))
        .unwrap_or_else(|| panic!("{summary}"));
    assert_eq!(
        row.split_whitespace().skip(2).collect::<Vec<_>>(),
        ["N/A", "N/A"],
        "{row}"
    );

    let mut moved = stale.clone();
    moved.theta[k] = 2.9;
    moved.packed_estimate = None;
    let mut edited = f1b_start(&p, Some(&F1B_UPPER_OPT));
    edited.theta = moved.theta.clone();
    assert!(
        crate::api::profiled_bound_report(&p.parsed.model, &p.population, &edited, true)
            .0
            .is_empty(),
        "the edited estimates name no level on a bound"
    );
    let out = replay(&moved);
    reports_held(&out, "(c)");
    // #2419: the reported estimates are the re-solved levels the step ran at, so the held
    // level's estimate is its bound, in the YAML next to `at_bound: true`. Only levels
    // move: every other θ, the objective and the EBEs are the incoming fit's.
    assert_eq!(out.theta[k].to_bits(), 3.0f64.to_bits());
    for i in (0..out.theta.len()).filter(|&i| !p.parsed.model.theta_names[i].contains('[')) {
        assert_eq!(out.theta[i].to_bits(), moved.theta[i].to_bits(), "{i}");
    }
    assert_eq!(out.ofv.to_bits(), moved.ofv.to_bits());
    assert!(out
        .subjects
        .iter()
        .zip(&moved.subjects)
        .all(|(a, b)| a.eta == b.eta));
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.yaml");
    crate::io::output::write_estimates_yaml(&out, path.to_str().unwrap()).unwrap();
    let yaml = std::fs::read_to_string(&path).unwrap();
    assert!(
        yaml.contains("  PLACEBO[STUDY=3,TIME=0]:\n    estimate: 3.000000\n    at_bound: true\n"),
        "{yaml}"
    );
    let resolved = |out: &crate::types::FitResult| -> Vec<String> {
        out.warnings
            .iter()
            .filter(|w| w.starts_with("W_COV_PROFILED_LEVELS_RESOLVED"))
            .cloned()
            .collect()
    };
    // Measured: the edit moves study 3's three free levels, no other study's. Each sentence
    // is asserted, so deleting any one of them fails here.
    let head = "W_COV_PROFILED_LEVELS_RESOLVED: the covariance step re-profiled 3 of the 18 \
                profiled levels away from the estimates run_covariance was handed; the largest \
                move is PLACEBO[STUDY=3,TIME=0], from 2.900000 to 3.000000 (+1.00e-1).";
    let tail = " The objective function value and the EBEs are still the incoming fit's.";
    assert_eq!(
        resolved(&out),
        [format!(
            "{head} The reported estimates are these re-solved levels, the point the \
             covariance matrix and the standard errors are computed at.{tail}"
        )]
    );
    assert_eq!(
        crate::types::classify_warning(&resolved(&out)[0]).category,
        crate::types::WarningCode::CovarianceStep
    );
    // The other side of the matrix gate: the held re-solve starved behind the guard
    // (#2420), the step fails after the first re-solve moved the levels. They are still
    // reported where it put them, and the warning says the step failed there.
    let starved = |t: String| bx(t).replace("TVE0", STARVE_HELD_RESOLVE);
    let (ps, fs) = f1b_evaluated(
        &starved,
        Some(&F1B_UPPER_OPT),
        false,
        CovarianceMethod::Hessian,
    );
    let mut moved_s = fs.clone();
    moved_s.theta[k] = 2.9;
    moved_s.packed_estimate = None;
    let failed = crate::run_covariance(
        &moved_s,
        Some(&ps.parsed.model),
        Some(&ps.population),
        &FitOptions {
            max_unconverged_frac: 0.0,
            ..o.clone()
        },
    )
    .unwrap_or_else(|e| panic!("{e}"));
    assert!(failed.covariance_matrix.is_none(), "{:?}", failed.warnings);
    assert!(
        failed.warnings.iter().any(|w| w.contains("EBE guard")),
        "{:?}",
        failed.warnings
    );
    assert_eq!(failed.theta[k].to_bits(), 3.0f64.to_bits());
    assert_eq!(
        resolved(&failed),
        [format!(
            "{head} The reported estimates are these re-solved levels; the covariance step \
             failed there and reports no standard errors.{tail}"
        )]
    );
    // A replay of the replayed fit moves nothing and drops the warning it carried.
    assert!(resolved(&replay(&out)).is_empty());
    // Cells (a) and (b) handed the step its own β̂: nothing moved, no warning, θ bit-equal.
    for (cell, f) in [("(a)", &fit), ("(b)", &stale)] {
        let r = replay(f);
        assert!(
            !r.warnings
                .iter()
                .any(|w| w.starts_with("W_COV_PROFILED_LEVELS_RESOLVED")),
            "{cell}: {:?}",
            r.warnings
        );
        assert!(
            r.theta
                .iter()
                .zip(&f.theta)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{cell}"
        );
    }
}

/// B2 (#2112), at `fit()` level: on F1 a box no level reaches leaves the whole fit — the
/// outer loop's evaluations, the OFV, every estimate, the covariance matrix and the
/// warnings — bit-identical to the unboxed declaration. A few outer evaluations, not a fit
/// to convergence: identity per evaluation is what the claim is, and the covariance step
/// re-profiles at every stencil point.
#[test]
fn an_unreached_box_leaves_the_fit_bit_identical() {
    let run = |edit: &dyn Fn(String) -> String| {
        let p = prep(F1, F1_DATA, Some(3), edit);
        let o = FitOptions {
            outer_maxiter: 2,
            run_covariance_step: true,
            covariance_method: CovarianceMethod::Hessian,
            ..options(&p)
        };
        crate::api::fit(&p.parsed.model, &p.population, &p.init_params, &o)
            .unwrap_or_else(|e| panic!("{e}"))
    };
    let plain = run(&|t| t);
    let wide = run(&boxed("(0.0, -1e9, 1e9)"));
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert!(plain.ofv.is_finite() && plain.n_iterations > 1);
    let cov = plain.covariance_matrix.as_ref().expect("the step ran");
    assert_eq!(plain.ofv.to_bits(), wide.ofv.to_bits());
    assert_eq!(plain.n_iterations, wide.n_iterations);
    assert_eq!(bits(&plain.theta), bits(&wide.theta));
    assert_eq!(
        bits(&plain.omega.iter().copied().collect::<Vec<_>>()),
        bits(&wide.omega.iter().copied().collect::<Vec<_>>())
    );
    assert_eq!(
        bits(cov.as_slice()),
        bits(wide.covariance_matrix.as_ref().unwrap().as_slice())
    );
    // The SS-equilibration diagnostic is collected process-wide, so a concurrently running
    // test's `SS=1` warnings can land in either fit (measured on the full `--lib` run);
    // F1 has no `SS` record, so none of them is this fit's.
    let own = |w: &[String]| -> Vec<String> {
        w.iter()
            .filter(|s| !s.starts_with("Steady-state (SS=1)"))
            .cloned()
            .collect()
    };
    assert_eq!(own(&plain.warnings), own(&wide.warnings));
}

// ── The covariance stencil's step: its own default unless set ────────────────────────

/// A profiled fit's covariance stencil steps at [`PROFILE_FD_HESSIAN_STEP`] while
/// `fd_hessian_step` is left alone, and at the user's value when they set one, including an
/// explicit `1e-2`, the global default's own value. The parser raises the flag with the key
/// and only then. The straddle: the profile default differs from the global one, so a
/// routing that ignored the flag would land on a different step either way.
///
/// Dies when the flag is ignored in either direction, when the parser does not set it, and
/// when it is set without the key. The call site, the `A_p` stencil, runs only off the
/// exact route since #2109, and is pinned there by
/// `the_profiled_covariance_step_runs_on_the_exact_blocks`.
#[test]
fn the_profiled_covariance_step_is_its_own_default_unless_set() {
    let unset = FitOptions::default();
    assert!(!unset.fd_hessian_step_set);
    assert_ne!(
        PROFILE_FD_HESSIAN_STEP, unset.fd_hessian_step,
        "the straddle"
    );
    assert_eq!(profile_fd_hessian_step(&unset), PROFILE_FD_HESSIAN_STEP);
    for h in [1e-2, 1e-3] {
        let set = FitOptions {
            fd_hessian_step: h,
            fd_hessian_step_set: true,
            ..FitOptions::default()
        };
        assert_eq!(profile_fd_hessian_step(&set), h, "explicit {h}");
    }

    let text = std::fs::read_to_string(root().join(F1)).unwrap();
    assert!(text.contains("  covariance = true\n"), "F1 moved");
    let parsed = |t: &str| {
        crate::parser::model_parser::parse_full_model(t)
            .unwrap_or_else(|e| panic!("{e}"))
            .fit_options
    };
    let without = parsed(&text);
    assert!(!without.fd_hessian_step_set, "no key, no flag");
    let with = parsed(&text.replace(
        "  covariance = true\n",
        "  covariance = true\n  fd_hessian_step = 1e-2\n",
    ));
    assert!(with.fd_hessian_step_set, "the key raises the flag");
    assert_eq!(profile_fd_hessian_step(&with), 1e-2);
}

// ── C2 / C4 / R2: the covariance step's exact blocks (#2109 PR4b) ──────────────────

/// `f` on a pool whose workers have the fit's stack (`FIT_RAYON_STACK_SIZE`): the
/// covariance Hessian's nested duals do not fit a default worker's. Thread-local switches
/// are read on the caller's thread, so set them inside `f`.
fn on_fit_pool<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .stack_size(crate::FIT_RAYON_STACK_SIZE)
        .build()
        .unwrap()
        .install(f)
}

/// The held template, the packed estimate with the solved levels, its free coordinates and
/// the solve at it, at the covariance step's inner tolerance.
struct CovPoint {
    hold: ModelParameters,
    x_hat: Vec<f64>,
    free: Vec<usize>,
    solve: ProfileSolve,
}

/// The covariance step's context for `p` (its inner tolerance, as `covariance` builds it).
fn cov_ctx<'a>(
    p: &'a crate::api::PreparedRun,
    layout: &'a ProfileLayout,
    o: &'a FitOptions,
) -> ProfileCtx<'a> {
    ProfileCtx {
        inner_tol: o
            .effective_cov_inner_tol(p.parsed.model.uses_closed_form_ltbs_inner())
            .min(PROFILE_INNER_TOL),
        ..ctx(p, layout, o)
    }
}

fn cov_point(p: &crate::api::PreparedRun, layout: &ProfileLayout, o: &FitOptions) -> CovPoint {
    let hold = hold_template(layout, &p.init_params);
    let PackedStart {
        packed: x, fixed, ..
    } = pack_with_bounds(&hold);
    let free: Vec<usize> = (0..x.len()).filter(|&i| !fixed[i]).collect();
    let solve = cov_ctx(p, layout, o)
        .solve(
            &unpack_params(&x, &hold),
            &ProfileWarm::from_template(layout, &p.init_params),
        )
        .unwrap_or_else(|e| panic!("the profile solve failed: {e:?}"));
    let x_hat = solved_point(layout, &hold, &x, &solve);
    CovPoint {
        hold,
        x_hat,
        free,
        solve,
    }
}

/// C2, per fixture: at `β̂`, every subject's exact `Dᵢ` and `Bᵢ` against the re-solved
/// differences they replace ([`ProfileCtx::subject_blocks_fd`]), each relative to the
/// largest entry of the subject's FD block. Returns the worst gaps of `D` and `B` and the
/// smallest of the subjects' largest FD entries (the straddle: a zero block agrees with
/// anything).
fn level_block_gaps(p: &crate::api::PreparedRun) -> (f64, f64, f64, f64) {
    let layout = layout_of(p);
    let o = options(p);
    let cp = cov_point(p, &layout, &o);
    let c = cov_ctx(p, &layout, &o);
    let (_, ls) = c
        .block_steps(&cp.hold, &cp.x_hat, &cp.free, &cp.solve)
        .unwrap();
    // The structural steps well below the stencil's default, whose truncation would be
    // most of the gap (`level_blocks_over_levels_are_the_theta_hessian`).
    let ts: Vec<f64> = cp
        .free
        .iter()
        .map(|&i| {
            let floor = fd_step_floor(&cp.hold, &c.model.default_params, i);
            fd_step(LEVEL_BLOCKS_ORACLE_STEP, cp.x_hat[i], floor)
        })
        .collect();
    // `Aᵢ`'s FD at the step production gives a declined subject (`level_blocks`).
    let a_step = profile_fd_hessian_step(&o);
    let a_steps: Vec<f64> = cp
        .free
        .iter()
        .map(|&i| {
            let floor = fd_step_floor(&cp.hold, &c.model.default_params, i);
            fd_step(a_step, cp.x_hat[i], floor)
        })
        .collect();
    let pt = crate::sens::ode_provider::adjoint::PackedPoint::new(&cp.hold, &cp.x_hat).unwrap();
    let per: Vec<(f64, f64, f64, f64)> = (0..p.population.subjects.len())
        .into_par_iter()
        .map(|i| {
            assert!(!cp.solve.off_mode[i], "subject {i} is off its mode");
            let (fd_d, fd_b) =
                c.subject_blocks_fd(i, &cp.hold, &cp.x_hat, &cp.free, &cp.solve, &ts, &ls);
            let fd_a =
                c.subject_structural_fd(i, &cp.hold, &cp.x_hat, &cp.free, &cp.solve, &a_steps);
            let (ad_a, ad_b, ad_d) = c
                .subject_blocks_adjoint(i, &pt, &cp.free, &cp.solve)
                .unwrap_or_else(|| panic!("subject {i}: the adjoint declined"));
            assert_eq!(ad_a.shape(), fd_a.shape());
            assert_eq!(ad_d.shape(), fd_d.shape());
            assert_eq!(ad_b.shape(), fd_b.shape());
            let gap = |ad: &DMatrix<f64>, fd: &DMatrix<f64>| {
                let scale = fd.iter().fold(0.0_f64, |a, v| {
                    assert!(v.is_finite(), "subject {i}: FD {v}");
                    a.max(v.abs())
                });
                let worst = ad.iter().zip(fd.iter()).fold(0.0_f64, |a, (x, y)| {
                    assert!(x.is_finite(), "subject {i}: adjoint {x}");
                    a.max((x - y).abs())
                });
                (worst / scale, scale)
            };
            let ((gd, sd), (gb, sb), (ga, sa)) =
                (gap(&ad_d, &fd_d), gap(&ad_b, &fd_b), gap(&ad_a, &fd_a));
            (gd, gb, ga, sd.min(sb).min(sa))
        })
        .collect();
    per.iter().fold(
        (0.0, 0.0, 0.0, f64::INFINITY),
        |(d, b, a, s), &(gd, gb, ga, si)| (d.max(gd), b.max(gb), a.max(ga), s.min(si)),
    )
}

/// C2. Every G2 fixture is on the adjoint and its `Dᵢ`, `Bᵢ` match the re-solved
/// differences. Dies when the block is taken along the packed axes without the EBE's
/// response (the held-EBE Hessian), when the ×2 to the OFV is dropped, and when the
/// slack's `−1` is dropped from `M` (the global fixture only, so it is required).
#[test]
fn the_exact_level_blocks_are_the_reconverged_fd_ones() {
    on_fit_pool(|| {
        for (name, p) in g2_fixtures() {
            let layout = layout_of(&p);
            let o = options(&p);
            assert!(
                ctx(&p, &layout, &o).cov_is_analytic(),
                "{name}: the fixture's covariance step runs on the adjoint"
            );
            let (gd, gb, ga, size) = level_block_gaps(&p);
            eprintln!(
                "C2 {name}: worst D {gd:.3e}, B {gb:.3e}, A {ga:.3e} (relative), smallest \
                 block {size:.3e}"
            );
            assert!(size > 0.0, "{name}: a block is identically zero");
            assert!(gd < C2_BOUND, "{name}: D {gd:.3e}");
            assert!(gb < C2_BOUND, "{name}: B {gb:.3e}");
            assert!(ga < C2_A_BOUND, "{name}: A {ga:.3e}");
        }
    });
}

/// Worst realised (Linux, macOS alike), relative to each subject's largest FD entry, structural
/// step `LEVEL_BLOCKS_ORACLE_STEP`: `D` 2.0e-6 (global), `B` 4.0e-5 (F2 car1); F1, linear
/// in η, 1.5e-11 / 1.4e-7. The FD side's truncation and its EBE noise; 2.5× headroom on `B`.
const C2_BOUND: f64 = 1e-4;
/// A declined subject's FD `Aᵢ` (`ProfileCtx::subject_structural_fd`, at the step
/// production gives it) against the adjoint's, relative to the largest entry. Worst
/// realised 6.9e-4 (nested), 2.2e-4–2.4e-4 on the others (macOS arm64): truncation, falling
/// as the step squared (2.4e-3 at 1e-2, 2.4e-5 at 1e-3, 6.1e-6 at 5e-4). 2.9× headroom.
const C2_A_BOUND: f64 = 2e-3;

/// C4: on F2 and the global fixture, the Schur `A_p = A − BD⁻¹Bᵀ` of the exact blocks
/// against the stencil of the profile objective it replaces (`fd_ofv_hessian`, the levels
/// re-profiled at every point), at a step below the stencil's default so its truncation
/// does not dominate (`C4_STEP`). Correlation scale, `|Δ_ij| / sqrt(S_ii S_jj)`. The straddle: `A`
/// alone (the levels held) is far from it, so a dropped `− BD⁻¹Bᵀ` dies.
#[test]
fn the_schur_profile_hessian_is_the_profile_stencil() {
    on_fit_pool(|| {
        for (name, p) in [
            ("F2", prep(F2, F2_DATA, Some(2), |t| t)),
            ("global", global_prep(2)),
        ] {
            let layout = layout_of(&p);
            let o = options(&p);
            let cp = cov_point(&p, &layout, &o);
            let c = cov_ctx(&p, &layout, &o);
            let blocks = c
                .level_blocks(&cp.hold, &cp.x_hat, &cp.free, &cp.solve)
                .unwrap();
            assert!(
                blocks.fd_subjects.is_empty(),
                "{name}: {:?}",
                blocks.fd_subjects
            );
            let a = blocks.a.as_ref().expect("every subject exact");
            let a_p = schur_profile_hessian(&layout, a, &blocks.d_groups, &blocks.b).unwrap();
            let warm = ProfileWarm::from_solve(&layout, &cp.solve);
            let f = |xv: &[f64]| {
                c.solve(&unpack_params(xv, &cp.hold), &warm)
                    .map_or(f64::NAN, |s| s.ofv)
            };
            let floor = fd_step_floors(&cp.hold, &c.model.default_params);
            let st = fd_ofv_hessian(
                cp.x_hat.len(),
                &cp.x_hat,
                &cp.free,
                &floor,
                C4_STEP,
                f(&cp.x_hat),
                &f,
                &o,
            )
            .unwrap();
            let nf = cp.free.len();
            let s = DMatrix::from_fn(nf, nf, |i, j| st.hess[(cp.free[i], cp.free[j])]);
            let corr_gap = |m: &DMatrix<f64>| {
                let mut worst = 0.0_f64;
                for i in 0..nf {
                    for j in 0..nf {
                        let g = (m[(i, j)] - s[(i, j)]).abs() / (s[(i, i)] * s[(j, j)]).sqrt();
                        assert!(g.is_finite(), "{name} ({i}, {j})");
                        worst = worst.max(g);
                    }
                }
                worst
            };
            let (schur, held) = (corr_gap(&a_p), corr_gap(a));
            eprintln!("C4 {name}: Schur vs stencil {schur:.3e}; A alone {held:.3e}");
            assert!(schur < C4_BOUND, "{name}: {schur:.3e}");
            assert!(held > C4_STRADDLE, "{name}: A alone is {held:.3e} from A_p");
        }
    });
}

/// The stencil's step. Its gap to the Schur complement is its own truncation, falling as the
/// step squared to a noise floor (F2 / global; Linux aarch64 and macOS arm64 alike at
/// 1e-3): 2.1e-2 / – at 1e-2,
/// 1.7e-3 / 3.3e-3 at 3e-3, 1.9e-4 / 3.7e-4 at 1e-3, 1.3e-4 / 5.7e-4 at 5e-4. The nested
/// fixture (`nested_corr`, left out: ~4 min a run in debug) converges the same way, with a
/// higher floor: 1.4e-1 / 1.9e-2 / 5.5e-3 / 2.9e-3 at 1e-2 / 5e-3 / 3e-3 / 2e-3, then
/// 1.7e-2 at 1e-3, where the re-solved objectives' noise takes over (macOS arm64).
const C4_STEP: f64 = 1e-3;
/// Worst realised 3.7e-4 (global) at `C4_STEP`; 4× headroom.
const C4_BOUND: f64 = 1.5e-3;
/// `A` alone is 0.47 (F2) and 0.54 (global) from the stencil; 2.3× below the smaller.
const C4_STRADDLE: f64 = 0.2;

/// `p`'s covariance step at its solved template, through `covariance_at_estimate` (the
/// path `fit()` and `run_covariance` share), with the count of steps whose `A_p` was the
/// Schur complement.
fn cov_step(p: &crate::api::PreparedRun, o: &FitOptions) -> (CovStepOutcome, usize) {
    let layout = layout_of(p);
    let c = ctx(p, &layout, o);
    let s = c
        .solve(
            &p.init_params,
            &ProfileWarm::from_template(&layout, &p.init_params),
        )
        .unwrap();
    let hold = hold_template(&layout, &p.init_params);
    let PackedStart {
        packed: x, fixed, ..
    } = pack_with_bounds(&hold);
    let free: Vec<usize> = (0..x.len()).filter(|&i| !fixed[i]).collect();
    let before = PROFILE_COV_SCHUR.with(std::cell::Cell::get);
    let out = c
        .covariance_at_estimate(&hold, &x, &free, &s.params, &s.etas, &s.kappas)
        .0;
    (out, PROFILE_COV_SCHUR.with(std::cell::Cell::get) - before)
}

/// A subject whose mode may not be one takes the re-solved differences for its level
/// blocks and its `Aᵢ`, and is listed, `A` still built (the adjoint's identity is exact only at a
/// mode, #2401 review row 2). At `inner_maxiter = 2` every F2 subject's solve stops short;
/// at F2's own setting none does: the straddle. Dies when the blocks ignore
/// `ProfileSolve::off_mode`.
#[test]
fn a_subject_off_its_mode_takes_the_resolved_level_blocks() {
    on_fit_pool(|| {
        let p = prep(F2, F2_DATA, Some(2), |t| t);
        let layout = layout_of(&p);
        let o = options(&p);
        let short = FitOptions {
            inner_maxiter: 2,
            ..o.clone()
        };
        for (opts, all_off) in [(&o, false), (&short, true)] {
            let cp = cov_point(&p, &layout, opts);
            assert!(
                cp.solve.off_mode.iter().all(|&b| b == all_off),
                "{:?}",
                cp.solve.off_mode
            );
            let blocks = cov_ctx(&p, &layout, opts)
                .level_blocks(&cp.hold, &cp.x_hat, &cp.free, &cp.solve)
                .unwrap();
            let want: Vec<usize> = if all_off {
                (0..p.population.subjects.len()).collect()
            } else {
                Vec::new()
            };
            assert_eq!(blocks.fd_subjects, want);
            assert!(blocks
                .a
                .as_ref()
                .is_some_and(|a| a.iter().all(|v| v.is_finite())));
        }
    });
}

/// A subject off its mode on fewer observations than `min_obs_for_convergence_check` keeps
/// the re-solved differences but is not named by the FD-fallback warnings (#2408). At `inner_maxiter = 2` every F2 subject (8 observations) stops
/// short; the straddle is the threshold, 8 (none excused) against 9 (all excused). Dies
/// when the excuse is dropped (the warnings name them at 9), widened past the threshold
/// (they are silent at 8), or applied to the fallback itself (the level blocks stop being
/// re-solved).
#[test]
fn an_off_mode_subject_below_min_obs_is_not_reported() {
    on_fit_pool(|| {
        let p = prep(F2, F2_DATA, Some(2), |t| t);
        let layout = layout_of(&p);
        let n = p.population.subjects.len();
        let all: Vec<usize> = (0..n).collect();
        assert!(p
            .population
            .subjects
            .iter()
            .all(|s| s.observations.len() == 8));
        for (min_obs, excused) in [(8, false), (9, true)] {
            let o = FitOptions {
                inner_maxiter: 2,
                min_obs_for_convergence_check: min_obs,
                ..options(&p)
            };
            let cp = cov_point(&p, &layout, &o);
            assert!(
                cp.solve.off_mode.iter().all(|&b| b),
                "{:?}",
                cp.solve.off_mode
            );
            assert!(
                cp.solve.off_mode_excused.iter().all(|&b| b == excused),
                "min_obs {min_obs}: {:?}",
                cp.solve.off_mode_excused
            );
            let want: Vec<usize> = if excused { Vec::new() } else { all.clone() };
            assert_eq!(cp.solve.fd_gradient_subjects, want, "min_obs {min_obs}");
            let c = cov_ctx(&p, &layout, &o);
            let blocks = c
                .level_blocks(&cp.hold, &cp.x_hat, &cp.free, &cp.solve)
                .unwrap();
            assert_eq!(blocks.fd_subjects, all, "the fallback itself stays");
            let out = c.exact_covariance(&cp.hold, &cp.x_hat, &cp.free, &cp.solve, Instant::now());
            let named: Vec<&String> = out
                .warnings
                .iter()
                .filter(|w| w.contains("exact analytic Hessian of their objective terms"))
                .collect();
            assert_eq!(named.is_empty(), excused, "{:?}", out.warnings);
            if !excused {
                assert!(
                    named[0].starts_with(&format!("{n} of {n} subjects")),
                    "{}",
                    named[0]
                );
            }
        }
    });
}

/// The fit's own warning across its evaluations (#2408), on the same straddle as
/// [`an_off_mode_subject_below_min_obs_is_not_reported`].
#[test]
fn a_profiled_fit_does_not_report_off_mode_subjects_below_min_obs() {
    let p = prep(F2, F2_DATA, Some(2), |t| t);
    for (min_obs, excused) in [(8, false), (9, true)] {
        let o = FitOptions {
            outer_maxiter: 2,
            inner_maxiter: 2,
            min_obs_for_convergence_check: min_obs,
            run_covariance_step: false,
            ..options(&p)
        };
        let r = optimize_profiled(&p.parsed.model, &p.population, &p.init_params, &o);
        let named = r
            .warnings
            .iter()
            .any(|w| w.contains("gradient of the profile solve during this fit"));
        assert_eq!(named, !excused, "min_obs {min_obs}: {:?}", r.warnings);
    }
}

/// R2's step values, each used by no other test: the stencil-call log is keyed by value.
const R2_EXACT_STEP: f64 = 1.3579e-2;
const R2_UNSET_STEP: f64 = 1.2345e-2;
/// The mixed step against the all-exact one: worst realised 1.6e-3 (macOS arm64), the
/// declined subject's FD `Aᵢ` at `PROFILE_FD_HESSIAN_STEP` and `Bᵢ` at the global step
/// (`level_blocks`' sweep). 2.5× headroom. With the subject's `Aᵢ` zeroed (measured) `A_p`
/// is no longer positive definite and the step fails, which the warning check catches.
const R2_MIXED_GAP: f64 = 4e-3;

/// R2, the step's engine. On F2 `A_p` is the Schur complement and nothing is warned, and a
/// second step from the same estimate is the first bit for bit (what the replay tests rely
/// on: both their calls take this route). `gradient = fd` and `analytic_cov_hessian =
/// false` each keep it on the stencil, which steps at `PROFILE_FD_HESSIAN_STEP` while
/// `fd_hessian_step` is unset (#2414); the exact route runs no stencil. A subject declined
/// at runtime takes FD for its own `Aᵢ`, `Bᵢ`, `Dᵢ`, keeps `A_p` a Schur complement close
/// to the all-exact one (#2417 review rows 3–4), and is warned about in full — on the
/// adjoint route only: the same switch under `gradient = fd` warns nothing. Dies when
/// either option is ignored, when a decline is not warned or is warned on the FD route,
/// when a decline sends `A_p` to the stencil or drops the subject's blocks, and when a
/// sentence of the warning is lost.
#[test]
fn the_profiled_covariance_step_runs_on_the_exact_blocks() {
    use crate::estimation::covariance::reconverge_log::take_fd_hessian;
    on_fit_pool(|| {
        let p = prep(F2, F2_DATA, Some(2), |t| t);
        let o = options(&p);
        // A step value no other test uses, so the stencil-call log is this test's alone.
        let exact_step = FitOptions {
            fd_hessian_step: R2_EXACT_STEP,
            fd_hessian_step_set: true,
            ..o.clone()
        };
        let (first, n1) = cov_step(&p, &exact_step);
        assert_eq!(
            take_fd_hessian(R2_EXACT_STEP),
            0,
            "the exact route ran the stencil"
        );
        let (second, n2) = cov_step(&p, &o);
        assert_eq!((n1, n2), (1, 1), "{:?}", first.warnings);
        assert!(first.warnings.is_empty(), "{:?}", first.warnings);
        let (m1, m2) = (first.matrix.unwrap(), second.matrix.unwrap());
        assert!(m1
            .iter()
            .zip(m2.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()));

        // Off the exact route the stencil steps at the profile's own default when
        // `fd_hessian_step` is left unset (#2414), whatever value the field holds.
        let off = FitOptions {
            analytic_cov_hessian: false,
            fd_hessian_step: R2_UNSET_STEP,
            fd_hessian_step_set: false,
            ..o.clone()
        };
        let (out, n) = cov_step(&p, &off);
        assert_eq!(
            n, 0,
            "analytic_cov_hessian = false ran the Schur complement"
        );
        assert_eq!(
            take_fd_hessian(R2_UNSET_STEP),
            0,
            "an unset fd_hessian_step reached the stencil"
        );
        assert!(
            take_fd_hessian(PROFILE_FD_HESSIAN_STEP) >= 1,
            "no stencil ran"
        );
        assert!(
            out.matrix.is_some() && out.warnings.is_empty(),
            "{:?}",
            out.warnings
        );
        let pf = prep(F2, F2_DATA, Some(2), forced_fd);
        let of = options(&pf);
        let (out, n) = cov_step(&pf, &of);
        assert_eq!(n, 0, "gradient = fd ran the Schur complement");
        assert!(
            out.matrix.is_some() && out.warnings.is_empty(),
            "{:?}",
            out.warnings
        );

        FORCE_PROFILE_COV_DECLINE.with(|c| c.set(Some(1)));
        let declined = cov_step(&p, &o);
        let declined_fd = cov_step(&pf, &of);
        FORCE_PROFILE_COV_DECLINE.with(|c| c.set(None));
        let (out, n) = declined;
        assert_eq!(n, 1, "a declined subject sent A_p off the Schur complement");
        let id = &p.population.subjects[1].id;
        let want = format!(
            "1 of {} subjects could not be given the exact analytic Hessian of their objective \
             terms in the covariance step (e.g. subject {id}). Those terms' Hessians used \
             reconverged finite differences, which are correct but slower.",
            p.population.subjects.len()
        );
        assert_eq!(out.warnings, vec![want]);
        // The mixed step (exact blocks for every subject but one, re-solved FD for that
        // one's `Aᵢ`, `Bᵢ`, `Dᵢ`, `A_p` still their Schur complement) against the all-exact
        // one, on the correlation scale (#2417 review rows 3–4): the gap is that one
        // subject's FD truncation.
        let mixed = out.matrix.expect("the mixed step publishes a matrix");
        let gap = (0..m1.nrows())
            .flat_map(|i| (0..m1.ncols()).map(move |j| (i, j)))
            .filter(|&(i, j)| m1[(i, i)] > 0.0 && m1[(j, j)] > 0.0)
            .map(|(i, j)| (mixed[(i, j)] - m1[(i, j)]).abs() / (m1[(i, i)] * m1[(j, j)]).sqrt())
            .fold(0.0_f64, |a, g| {
                assert!(g.is_finite());
                a.max(g)
            });
        eprintln!("R2 mixed vs exact: worst correlation-scale gap {gap:.3e}");
        assert!(gap < R2_MIXED_GAP, "{gap:.3e}");
        let (out, n) = declined_fd;
        assert_eq!(n, 0);
        assert!(
            out.warnings.is_empty(),
            "the FD route warned: {:?}",
            out.warnings
        );
    });
}

/// The exact route's matrix by hand from the estimate's solve by `solver`, as
/// `covariance_at_estimate` builds it: the solve, then the blocks and the Schur `A_p` at
/// its modes, both in the covariance step's context.
fn exact_step_by_hand(p: &crate::api::PreparedRun, solver: &ProfileCtx<'_>) -> DMatrix<f64> {
    let layout = solver.layout;
    let c = solver.at_cov_tol();
    let s = ctx(p, layout, solver.options)
        .solve(
            &p.init_params,
            &ProfileWarm::from_template(layout, &p.init_params),
        )
        .unwrap();
    let hold = hold_template(layout, &p.init_params);
    let PackedStart {
        packed: x, fixed, ..
    } = pack_with_bounds(&hold);
    let free: Vec<usize> = (0..x.len()).filter(|&i| !fixed[i]).collect();
    let warm = ProfileWarm {
        beta: layout.coords.iter().map(|&k| s.params.theta[k]).collect(),
        etas: Some(joint_warm_starts(&s.etas, &s.kappas)),
    };
    let sol = solver.solve(&s.params, &warm).unwrap();
    let x_hat = solved_point(layout, &hold, &x, &sol);
    let blocks = c.level_blocks(&hold, &x_hat, &free, &sol).unwrap();
    let a_p = schur_profile_hessian(
        layout,
        blocks.a.as_ref().unwrap(),
        &blocks.d_groups,
        &blocks.b,
    )
    .unwrap();
    c.covariance_from_profile_hessian(
        &hold,
        &x_hat,
        &free,
        &sol,
        &a_p,
        Instant::now(),
        Some(blocks),
    )
    .matrix
    .unwrap()
}

/// #2417 review row 1: the exact route reads its blocks at a solve run at the covariance
/// step's inner tolerance, not the fit's. With `cov_inner_tol = 1e-12` against the solve's
/// `PROFILE_INNER_TOL = 1e-7` (by default the two are equal, which is how the reviewer's
/// mutation passed every test), the step's matrix is the by-hand one at `1e-12` bit for
/// bit, and differs from the by-hand one at `1e-7`: the straddle. Measured on F2 (macOS
/// arm64), warm-started from the fit's modes: identical bits down to `1e-10`, 8.5e-13
/// apart at `1e-12` with every subject still on its mode; from `1e-13` a subject's solve
/// falls back to Nelder–Mead and declines to FD. Dies when the estimate's solve runs at
/// the fit's tolerance.
#[test]
fn the_exact_covariance_step_reads_its_blocks_at_the_cov_tolerance() {
    on_fit_pool(|| {
        let p = prep(F2, F2_DATA, Some(2), |t| t);
        let layout = layout_of(&p);
        let o = FitOptions {
            cov_inner_tol: Some(1e-12),
            ..options(&p)
        };
        let c = ctx(&p, &layout, &o);
        assert!(
            c.at_cov_tol().inner_tol < c.inner_tol,
            "the fixture separates the two"
        );
        let (out, n) = cov_step(&p, &o);
        assert_eq!(n, 1, "{:?}", out.warnings);
        let got = out.matrix.unwrap();
        let bits = |m: &DMatrix<f64>| m.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&got), bits(&exact_step_by_hand(&p, &c.at_cov_tol())));
        assert_ne!(
            bits(&got),
            bits(&exact_step_by_hand(&p, &c)),
            "the fit's tolerance gives the same matrix: the straddle is gone"
        );
    });
}

/// The held-layout path (#2112, merged into #2417 at rebase) reads its blocks at the step's
/// tolerance too: with `PLACEBO[STUDY=3,TIME=0]` on its upper bound (F1, the `C4`/`W1`
/// fixture of `a_fit_with_a_level_on_its_bound_holds_it_in_the_covariance_step`) and
/// `cov_inner_tol = 1e-12`, every inner solve of the step ran at `1e-12`, the held
/// re-solve's included; the one held level shows the held path ran. Dies when the held
/// context is built from the fit's tolerance (`..*self`).
#[test]
fn the_held_layout_resolve_runs_at_the_cov_tolerance() {
    use crate::estimation::inner_optimizer::record_inner_solves;
    use crate::estimation::inner_optimizer::test_helpers::on_one_thread;
    let p = prep(F1, F1_DATA, None, boxed("(0.0, -10.0, 3.0)"));
    let layout = layout_of(&p);
    let start = f1b_start(&p, Some(&F1B_UPPER_OPT));
    let o = FitOptions {
        cov_inner_tol: Some(1e-12),
        covariance_method: CovarianceMethod::Hessian,
        ..options(&p)
    };
    let c = ctx(&p, &layout, &o);
    assert!(c.cov_is_analytic());
    let s = c
        .solve(&start, &ProfileWarm::from_template(&layout, &start))
        .unwrap();
    let hold = hold_template(&layout, &start);
    let PackedStart {
        packed: x, fixed, ..
    } = pack_with_bounds(&hold);
    let free: Vec<usize> = (0..x.len()).filter(|&i| !fixed[i]).collect();
    let ((out, pinned), rec) = on_one_thread(|| {
        record_inner_solves(|| {
            c.covariance_at_estimate(&hold, &x, &free, &s.params, &s.etas, &s.kappas)
        })
    });
    assert_eq!(
        pinned.map(|r| r.pinned.len()),
        Some(1),
        "one level held: {:?}",
        out.warnings
    );
    assert!(out.matrix.is_some(), "{:?}", out.warnings);
    let tols: Vec<f64> = rec.solve_tols.iter().map(|&(_, t)| t).collect();
    assert!(!tols.is_empty());
    assert!(
        tols.iter().all(|&t| t == 1e-12),
        "a solve off the step's tolerance: {tols:?}"
    );
}

/// #2420: the held-layout re-solve (#2112) goes through the #2214 EBE guard, as the first
/// re-solve does; on the exact route it is the solve every block is read from (#2417). With
/// `STARVE_HELD_RESOLVE` the held re-solve runs cold at `inner_maxiter = 0`, so its EBEs
/// stop short while the first re-solve's (warm, at their modes) do not. On F1 at the
/// upper-box optimum, `PLACEBO[STUDY=3,TIME=0]` on its bound, the step fails with the
/// guard's warning, its counts the held solve's, and still reports the level it held.
/// The straddle: the same fixture unboxed holds no level, never runs the held solve, and
/// passes the guard under the same switch. Dies when the held solve skips the guard.
#[test]
fn a_guarded_held_layout_resolve_fails_the_step() {
    let step = |edit: &dyn Fn(String) -> String| {
        let p = prep(F1, F1_DATA, None, |t| {
            edit(t).replace("TVE0", STARVE_HELD_RESOLVE)
        });
        let layout = layout_of(&p);
        let start = f1b_start(&p, Some(&F1B_UPPER_OPT));
        let o = FitOptions {
            covariance_method: CovarianceMethod::Hessian,
            max_unconverged_frac: 0.0,
            ..options(&p)
        };
        let c = ctx(&p, &layout, &o);
        let s = c
            .solve(&start, &ProfileWarm::from_template(&layout, &start))
            .unwrap();
        let hold = hold_template(&layout, &start);
        let PackedStart {
            packed: x, fixed, ..
        } = pack_with_bounds(&hold);
        let free: Vec<usize> = (0..x.len()).filter(|&i| !fixed[i]).collect();
        let (out, resolved) =
            c.covariance_at_estimate(&hold, &x, &free, &s.params, &s.etas, &s.kappas);
        (p.population.subjects.len(), out, resolved)
    };
    let (n, held, resolved) = step(&boxed("(0.0, -10.0, 3.0)"));
    assert_eq!(resolved.map(|r| r.pinned.len()), Some(1), "one level held");
    assert!(held.matrix.is_none());
    assert_eq!(held.warnings.len(), 1, "{:?}", held.warnings);
    // The text names the held re-solve, not the first one (#2453 review row 4): the
    // first re-solve passed, and a failure should say which of the two fired.
    let w = &held.warnings[0];
    assert!(
        w.starts_with(&format!(
            "Covariance step failed: the profile re-solve with the levels on their bound held \
             there left {n} of {n} subjects' EBEs unconverged and rejected 0 at their inner \
             start."
        )),
        "{w}"
    );
    assert!(
        w.contains(
            " The EBE guard rejects an unconverged fraction above max_unconverged_frac = 0."
        ),
        "{w}"
    );

    let (_, unheld, resolved) = step(&|t| t);
    assert_eq!(resolved.map(|r| r.pinned.len()), Some(0), "no level held");
    assert!(
        !unheld.warnings.iter().any(|w| w.contains("EBE guard")),
        "{:?}",
        unheld.warnings
    );
}

/// `W_COV_PROFILED_LEVELS_RESOLVED` names the largest move by its signed size, so a move
/// under the six printed decimals still reads as one (#2453 review row 2): the trigger is a
/// bit difference, and "from 1.500000 to 1.500000" alone says nothing moved. Hand-set
/// levels, since no fixture's re-solve moves a level that little: two move (one by +1e-9,
/// one by −3e-9, the larger), one does not and is not counted; the moved ones are written
/// into θ. Nothing moved: `None`, θ untouched.
#[test]
fn the_resolved_levels_warning_signs_a_move_below_its_printed_decimals() {
    let p = prep(F1, F1_DATA, None, |t| t);
    let mut params = p.init_params.clone();
    let (a, b, c) = (4, 5, 6);
    for k in [a, b, c] {
        params.theta[k] = 1.5;
    }
    let levels = [(a, 1.5 + 1e-9), (b, 1.5 - 3e-9), (c, 1.5)];
    let w = crate::api::report_resolved_levels(&levels, &mut params, true).unwrap();
    assert!(
        w.starts_with(&format!(
            "W_COV_PROFILED_LEVELS_RESOLVED: the covariance step re-profiled 2 of the 3 \
             profiled levels away from the estimates run_covariance was handed; the largest \
             move is {}, from 1.500000 to 1.500000 (-3.00e-9).",
            params.theta_names[b]
        )),
        "{w}"
    );
    assert_eq!(
        [a, b, c].map(|k| params.theta[k].to_bits()),
        [1.5 + 1e-9, 1.5 - 3e-9, 1.5f64].map(f64::to_bits)
    );
    let before = params.theta.clone();
    assert!(crate::api::report_resolved_levels(&levels, &mut params, true).is_none());
    assert_eq!(params.theta, before);
}

/// #2418: a profiled level the covariance step held on its bound (#2112) is held by the
/// uncertainty samplers too, as a FIX θ is. On F1 at the upper-box optimum, with the step
/// on (`PLACEBO[STUDY=3,TIME=0]` held at 3.0, its covariance row and column zero), every one
/// of 400 asymptotic draws (seed 7) and every retained SIR resample has the held level
/// bit-equal to its bound, while a free level of the same study varies: without the hold
/// the review measured 206/400 asymptotic draws above the bound, drawn at SD ≈ 1e-4 from
/// the regularised Cholesky. SIR samples the free subspace, so the level leaves it.
/// Measured with each hold removed in turn (macOS arm64): 400/400 asymptotic draws and
/// 100/100 SIR resamples off the bound's bits; with the hold, 0 and 0.
#[test]
fn uncertainty_draws_hold_a_level_the_step_held_on_its_bound() {
    use rand::SeedableRng;
    let bx = boxed("(0.0, -10.0, 3.0)");
    let (p, fit) = f1b_evaluated(&bx, Some(&F1B_UPPER_OPT), true, CovarianceMethod::Hessian);
    let k = theta_named(&p, "PLACEBO[STUDY=3,TIME=0]");
    let free = theta_named(&p, "PLACEBO[STUDY=3,TIME=4]");
    assert_eq!(fit.profiled_levels_at_bound, [k]);
    assert_eq!(fit.theta[k].to_bits(), 3.0f64.to_bits());
    let cov = fit
        .covariance_matrix
        .clone()
        .expect("the step held the level");
    assert_eq!(cov[(k, k)], 0.0);
    assert!(cov[(free, free)] > 0.0);
    let template =
        crate::estimation::uncertainty_samples::fitted_params_from_result(&fit, &p.parsed.model)
            .unwrap();

    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let draws = crate::estimation::uncertainty_samples::draw_parameter_samples(
        &fit,
        &template,
        400,
        crate::estimation::uncertainty_samples::UncertaintyMethod::Asymptotic,
        &mut rng,
    )
    .unwrap();
    let off = draws
        .iter()
        .filter(|d| d.theta[k].to_bits() != 3.0f64.to_bits())
        .count();
    let moved = draws
        .iter()
        .filter(|d| d.theta[free] != fit.theta[free])
        .count();
    eprintln!("asymptotic: held off {off}/400, free moved {moved}/400");
    assert_eq!(off, 0, "asymptotic draws off the held level's bound");
    assert_eq!(moved, 400, "the free level of the same study varies");

    let etas: Vec<DVector<f64>> = fit.subjects.iter().map(|s| s.eta.clone()).collect();
    let o = FitOptions {
        sir_samples: 200,
        sir_resamples: 100,
        sir_keep_samples: true,
        sir_seed: Some(7),
        ..options(&p)
    };
    let r = crate::estimation::sir::run_sir_core(
        &p.parsed.model,
        &p.population,
        &template,
        &etas,
        &cov,
        fit.ofv,
        &o,
    )
    .unwrap();
    let pool = r.resamples_packed.expect("kept");
    let x_hat = pack_with_bounds(&template).packed;
    let off = pool
        .iter()
        .filter(|x| x[k].to_bits() != x_hat[k].to_bits())
        .count();
    let moved = pool.iter().filter(|x| x[free] != x_hat[free]).count();
    eprintln!("sir: held off {off}/{}, free moved {moved}", pool.len());
    assert_eq!(pool.len(), 100);
    assert_eq!(off, 0, "SIR resamples off the held level's bound");
    assert!(
        moved > 50,
        "the free level of the same study varies: {moved}"
    );
}
