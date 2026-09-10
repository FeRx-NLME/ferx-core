//! What adaptive FD intervals actually buy on a noisy objective (#1314).
//!
//! Tier 3, and the end-to-end half of #1314's validation: the rest of the suite
//! pins the *interval search* on closed forms, and this file measures whether
//! the search moves an outer gradient on a real model.
//!
//! **The reference has to be each objective's own true gradient, not a
//! tight-tolerance one.** Comparing a loose-objective FD gradient against
//! `∇f_tight` sums two unrelated quantities: the solver *bias* `∇f_loose −
//! ∇f_tight`, a property of the objective that no stencil can remove, and the
//! *stencil* error `FD_h(f_loose) − ∇f_loose`, the only thing an interval
//! policy touches. The first cut of this test conflated them and read 8.985e-4
//! at `ode_reltol = 1e-4` against 8.755e-4 at `1e-6` — a 40× noise reduction
//! moving the number by 3%, the signature of a floor that is not noise. So each
//! row's reference is a wide-window least-squares slope of that same
//! objective, and the bias is reported beside it rather than folded in.
//!
//! The measured conclusion, which is a negative one and is why both tests
//! below exist: `examples/mm_iv.ferx` never enters the regime the adaptive
//! searches are for. Its OFV noise is ~3e-6 against gradients of 10–1500, and
//! the fixed `h ≈ 2e-4` stencil resolves that to ~1e-3 relative — so there is
//! nothing for a search to recover, and `gill` is actively worse. The second
//! test establishes the crossover on the same objective with controlled noise
//! added, so the condition is quantitative instead of anecdotal.

use super::*;
use crate::parser::model_parser::parse_full_model;
use nalgebra::DMatrix;

const MODEL: &str = include_str!("../../examples/mm_iv.ferx");

/// Points per least-squares probe window.
const PROBE_N: usize = 15;

/// The fixed stencil's interval, `central_diff_packed`'s `eps * (1 + |x|)`.
fn fixed_h(x: f64) -> f64 {
    1e-4 * (1.0 + x.abs())
}

/// One row of the sweep: an objective, and how it was made noisy.
struct Setting {
    label: &'static str,
    reltol: f64,
    abstol: f64,
    inner_tol: f64,
}

/// The model at a chosen accuracy. ODE tolerances are stamped onto the spec at
/// **parse time**, so they have to go through the model text — a `FitOptions`
/// field set after parsing does not reach a direct inner-loop call. `inner_tol`
/// is read from `FitOptions` at call time, so it is set on the struct.
fn fixture(s: &Setting) -> (CompiledModel, FitOptions, Population, ModelParameters) {
    let text = format!(
        "{MODEL}\n  ode_reltol = {:e}\n  ode_abstol = {:e}\n",
        s.reltol, s.abstol
    );
    let parsed = parse_full_model(&text).expect("mm_iv parses");
    let root = env!("CARGO_MANIFEST_DIR");
    let pop = crate::read_nonmem_csv(
        std::path::Path::new(&format!("{root}/data/mm_iv.csv")),
        None,
        None,
    )
    .expect("mm_iv data");
    let init = parsed.model.default_params.clone();
    let options = FitOptions {
        inner_tol: s.inner_tol,
        ..parsed.fit_options
    };
    (parsed.model, options, pop, init)
}

const REFERENCE: Setting = Setting {
    label: "reference   ",
    reltol: 1e-12,
    abstol: 1e-14,
    inner_tol: 1e-10,
};
const FERX_DEFAULT: Setting = Setting {
    label: "ferx default",
    reltol: 1e-4,
    abstol: 1e-6,
    inner_tol: 1e-5,
};

/// OFV at a packed point, re-solving the inner loop from a cold start.
///
/// Cold start is deliberate: a warm start carries state across calls, and an
/// FD stencil evaluated on something that is not a function of `x` alone
/// measures the history, not the derivative.
fn ofv_at(
    model: &CompiledModel,
    pop: &Population,
    init: &ModelParameters,
    options: &FitOptions,
    xv: &[f64],
) -> Option<f64> {
    let params = unpack_params(xv, init);
    let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
    let (ehs, hms, _stats, kappas) = run_inner_loop_warm_with_fd_config(
        model,
        pop,
        &params,
        options.inner_maxiter,
        options.inner_tol,
        None,
        Some(&mu_k),
        options.min_obs_for_convergence_check as usize,
        options.inner_restarts,
        InnerFdConfig::from_options(options),
    );
    let raw = 2.0 * pop_nll_opts(model, pop, &params, &ehs, &hms, &kappas, options);
    raw.is_finite().then_some(raw)
}

/// Deterministic pseudo-noise of amplitude `amp`, keyed on the whole packed
/// point.
///
/// Keyed on the bits so the perturbed objective stays a *function* of `x`: an
/// FD stencil evaluated on something that returns a fresh draw per call
/// measures the draw, not the derivative, and would make every policy look
/// equally bad.
fn ripple(xv: &[f64], amp: f64) -> f64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in xv {
        h ^= v.to_bits();
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let unit = (h >> 11) as f64 / ((1u64 << 53) as f64);
    amp * (2.0 * unit - 1.0)
}

/// Least-squares polynomial fit of `f` over `x0 ± half_width` in coordinate
/// `k`, returning `(slope at x0, max |residual|)`.
///
/// `degree` trades this estimate's two failure modes against each other: a
/// narrow window wants degree 2 (over a window that small any smooth function
/// is a parabola, so the residual is noise and nothing else), a wide window
/// wants degree 3 so real cubic curvature is not charged to the slope.
fn probe(
    f: &dyn Fn(&[f64]) -> Option<f64>,
    x: &[f64],
    k: usize,
    half_width: f64,
    degree: usize,
) -> (f64, f64) {
    let (mut ts, mut ys) = (Vec::new(), Vec::new());
    for i in 0..PROBE_N {
        let t = -half_width + 2.0 * half_width * (i as f64) / ((PROBE_N - 1) as f64);
        let mut xv = x.to_vec();
        xv[k] = x[k] + t;
        if let Some(y) = f(&xv) {
            ts.push(t);
            ys.push(y);
        }
    }
    assert!(
        ts.len() > degree + 3,
        "objective unscorable across the probe window"
    );
    let a = DMatrix::from_fn(ts.len(), degree + 1, |r, c| ts[r].powi(c as i32));
    let b = DMatrix::from_column_slice(ts.len(), 1, &ys);
    let coef = a
        .clone()
        .svd(true, true)
        .solve(&b, 1e-14)
        .expect("least squares");
    let fitted = a * &coef;
    let worst = (0..ys.len()).fold(0.0_f64, |w, i| {
        let r = ys[i] - fitted[(i, 0)];
        assert!(r.is_finite(), "non-finite residual");
        w.max(r.abs())
    });
    (coef[(1, 0)], worst)
}

/// The interval one adaptive search settles on for coordinate `k` — the same
/// call `central_diff_packed` makes, so the reported `h` is the one used.
fn chosen_interval(
    method: OuterFdMethod,
    x: &[f64],
    k: usize,
    bounds: &PackedBounds,
    noise: f64,
    eval: &dyn Fn(&[f64]) -> Option<f64>,
) -> Option<(f64, bool)> {
    let mut xw = x.to_vec();
    adaptive_first_derivative(
        method,
        x[k],
        fixed_h(x[k]),
        AxisBounds {
            lower: bounds.lower[k],
            upper: bounds.upper[k],
        },
        noise,
        |coordinate| {
            xw[k] = coordinate;
            eval(&xw)
        },
    )
    .map(|est| (est.h, est.accepted))
}

/// Worst relative error of `g` against `reference` over the free coordinates.
fn worst_rel_err(g: &[f64], reference: &[f64], free: &[usize], tag: &str) -> f64 {
    free.iter().enumerate().fold(0.0_f64, |w, (i, &k)| {
        assert!(g[k].is_finite(), "{tag} produced a non-finite gradient");
        w.max((g[k] - reference[i]).abs() / reference[i].abs().max(1e-8))
    })
}

/// Each policy's worst relative gradient error against `truth`, given a
/// measured `noise` bound to hand the adaptive searches.
fn policy_errors(
    x: &[f64],
    fixed_mask: &[bool],
    bounds: &PackedBounds,
    free: &[usize],
    truth: &[f64],
    noise: f64,
    eval: &dyn Fn(&[f64]) -> Option<f64>,
) -> Vec<f64> {
    [
        OuterFdMethod::Fixed,
        OuterFdMethod::Shi,
        OuterFdMethod::Gill,
    ]
    .into_iter()
    .map(|method| {
        let n = (method != OuterFdMethod::Fixed).then_some(noise);
        let g = central_diff_packed(x, fixed_mask, bounds, method, n, None, eval);
        worst_rel_err(&g, truth, free, &format!("{method:?}"))
    })
    .collect()
}

/// Shared setup: the packed point, its masks, and the free coordinates.
fn packed_setup(init: &ModelParameters) -> (Vec<f64>, Vec<bool>, PackedBounds, Vec<usize>) {
    let x = pack_params(init);
    let fixed_mask = packed_fixed_mask(init);
    let bounds = compute_bounds(init);
    let free: Vec<usize> = (0..x.len()).filter(|&k| !fixed_mask[k]).collect();
    (x, fixed_mask, bounds, free)
}

/// On a real ferx model at realistic tolerances, the fixed stencil is *not*
/// noise-limited — so the adaptive searches have nothing to recover.
///
/// This is a characterisation test recording a negative result. Its assertions
/// are the ones that are actually true of this fixture: the fixed stencil
/// resolves every setting to better than 1% (so a regression that breaks it
/// reddens this), and no policy returns a non-finite gradient. It deliberately
/// does **not** assert that `shi` wins, because on this model it does not.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: inner-loop sweeps at several tolerances, opt in with --features slow-tests"
)]
fn the_fixed_stencil_is_not_noise_limited_on_a_realistic_ode_model() {
    let settings = [
        REFERENCE,
        FERX_DEFAULT,
        Setting {
            label: "loose ode   ",
            reltol: 1e-3,
            abstol: 1e-5,
            inner_tol: 1e-5,
        },
        Setting {
            label: "loose inner ",
            reltol: 1e-4,
            abstol: 1e-6,
            inner_tol: 1e-3,
        },
        Setting {
            label: "loose both  ",
            reltol: 1e-3,
            abstol: 1e-5,
            inner_tol: 1e-2,
        },
    ];

    let (_, _, pop, init) = fixture(&REFERENCE);
    let (x, fixed_mask, bounds, free) = packed_setup(&init);
    let names = coordinate_names(&init);

    // The tight objective's gradient, used only to report each setting's bias.
    let (rm, ro, _, _) = fixture(&REFERENCE);
    let tight = |v: &[f64]| ofv_at(&rm, &pop, &init, &ro, v);
    let tight_grad: Vec<f64> = free
        .iter()
        .map(|&k| probe(&tight, &x, k, 1e-2 * (1.0 + x[k].abs()), 3).0)
        .collect();

    println!("\n=== reference gradient (ode_reltol 1e-12, inner_tol 1e-10) ===");
    for (i, &k) in free.iter().enumerate() {
        println!("{:18} {:14.6e}", names[k], tight_grad[i]);
    }

    println!("\n=== worst relative error vs each objective's OWN true gradient ===");
    println!("setting        measured noise    bias        fixed         shi        gill");

    let mut rows = Vec::new();
    for s in &settings {
        let (m, o, _, _) = fixture(s);
        let loose = |v: &[f64]| ofv_at(&m, &pop, &init, &o, v);

        // This objective's own smooth gradient, and its noise. Both measured.
        let mut truth = Vec::new();
        let mut noise = 0.0_f64;
        for &k in &free {
            let scale = 1.0 + x[k].abs();
            truth.push(probe(&loose, &x, k, 1e-2 * scale, 3).0);
            noise = noise.max(probe(&loose, &x, k, 1e-4 * scale, 2).1);
        }
        let bias = (0..free.len()).fold(0.0_f64, |w, i| {
            w.max((truth[i] - tight_grad[i]).abs() / tight_grad[i].abs().max(1e-8))
        });

        let err = policy_errors(&x, &fixed_mask, &bounds, &free, &truth, noise, &loose);
        println!(
            "{}   {noise:12.3e}  {bias:9.3e}  {:10.3e}  {:10.3e}  {:10.3e}",
            s.label, err[0], err[1], err[2]
        );
        rows.push((s.label, noise, err));
    }

    // ── What the searches chose at the noisiest setting. ──
    let worst = settings.last().expect("settings non-empty");
    let (m, o, _, _) = fixture(worst);
    let loose = |v: &[f64]| ofv_at(&m, &pop, &init, &o, v);
    let noise = rows.last().expect("rows non-empty").1;
    println!(
        "\n=== intervals chosen at '{}' (noise {noise:.3e}) ===",
        worst.label.trim()
    );
    println!("coordinate           fixed h        shi h  acc        gill h  acc");
    for &k in &free {
        print!("{:18} {:10.3e}", names[k], fixed_h(x[k]));
        for method in [OuterFdMethod::Shi, OuterFdMethod::Gill] {
            match chosen_interval(method, &x, k, &bounds, noise, &loose) {
                Some((h, acc)) => print!("  {h:11.3e} {:>4}", if acc { "yes" } else { "no" }),
                None => print!("  {:>11} {:>4}", "fallback", "-"),
            }
        }
        println!();
    }

    // The measured facts. Every setting's noise is orders of magnitude below
    // what the fixed interval resolves, so the fixed stencil stays inside 1%
    // throughout — the reason `shi` has no headroom to win here.
    for (label, noise, err) in &rows {
        assert!(
            *noise < 1e-4,
            "'{}' is noisier than this test assumes ({noise:.3e}); the crossover \
             test's premise may need revisiting",
            label.trim()
        );
        assert!(
            err[0] < 1e-2,
            "'{}' fixed-stencil error {:.3e} exceeds 1% — the default outer \
             gradient has regressed",
            label.trim(),
            err[0]
        );
    }
}

/// With controlled noise added to the same objective, the Shi search recovers
/// a gradient the fixed interval loses — and the amplitude where that starts
/// is the answer to "does this feature apply to my model".
///
/// The fixed stencil's noise-driven error is `≈ ε/h`, so relative to a gradient
/// `|g|` it matters once `ε ≳ |g|·h`. With `h ≈ 2e-4` that is `ε ≳ 2e-3` for
/// the smallest gradient here — which is ~1000× the noise the real objective
/// actually carries.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: repeated inner-loop gradients, opt in with --features slow-tests"
)]
fn adaptive_intervals_recover_the_gradient_once_noise_exceeds_the_fixed_interval() {
    let (m, o, pop, init) = fixture(&FERX_DEFAULT);
    let (x, fixed_mask, bounds, free) = packed_setup(&init);
    let clean = |v: &[f64]| ofv_at(&m, &pop, &init, &o, v);

    // The clean objective's own gradient. `ripple` is zero-mean and keyed on
    // the point, so it perturbs the objective without moving this.
    let truth: Vec<f64> = free
        .iter()
        .map(|&k| probe(&clean, &x, k, 1e-2 * (1.0 + x[k].abs()), 3).0)
        .collect();
    let smallest = truth
        .iter()
        .fold(f64::INFINITY, |w, g| w.min(g.abs()))
        .max(1e-12);
    println!(
        "\nsmallest |gradient| = {smallest:.4e}, fixed h ≈ {:.3e}",
        fixed_h(x[0])
    );
    println!(
        "predicted crossover at noise ≈ |g|·h ≈ {:.3e}",
        smallest * fixed_h(x[0])
    );

    println!("\n=== added noise amplitude vs worst relative gradient error ===");
    println!("amplitude        fixed         shi        gill");
    let mut rows = Vec::new();
    for amp in [1e-6, 1e-4, 1e-2, 1e0] {
        let noisy = |v: &[f64]| clean(v).map(|y| y + ripple(v, amp));
        let err = policy_errors(&x, &fixed_mask, &bounds, &free, &truth, amp, &noisy);
        println!(
            "{amp:9.0e}  {:10.3e}  {:10.3e}  {:10.3e}",
            err[0], err[1], err[2]
        );
        rows.push((amp, err));
    }

    let (amp, err) = rows.last().expect("rows non-empty");
    let (fixed_err, shi_err) = (err[0], err[1]);
    println!("\nat amplitude {amp:.0e}: fixed {fixed_err:.3e}, shi {shi_err:.3e}");
    // Straddle: the fixed stencil must genuinely have lost the gradient, or
    // "shi is better" would be a comparison between two correct answers.
    assert!(
        fixed_err > 1e-1,
        "fixed stencil is not noise-swamped at amplitude {amp:.0e} \
         (error {fixed_err:.3e}); this test cannot show what it claims"
    );
    assert!(
        shi_err < fixed_err / 10.0,
        "shi {shi_err:.3e} did not recover the gradient the fixed stencil lost \
         ({fixed_err:.3e})"
    );
}
