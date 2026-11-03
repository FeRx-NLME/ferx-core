//! Mini-batch (incremental) SAEM, `batch_fraction` (#2607).
//!
//! Two claims are pinned here. First, the default `batch_fraction = 1` is the
//! full-batch algorithm *bit for bit*: it draws no batch, so no RNG stream, no
//! counter and no adaptation step differs from the pre-#2607 loop. Second, a
//! fraction below one actually changes which chains move — a mini-batch test
//! that agrees with full batch is the fast-path trap `AGENTS.md` names, so the
//! `α < 1` arms assert that the run *differs* from full batch as well as that
//! it is sound.

use super::{batch_size, run_saem, saem_batch_mask, validate_batch_fraction};
use crate::parser::model_parser::{apply_fit_option, parse_model_string};
use crate::types::*;

const MODEL: &str = r"
[parameters]
  theta TVCL(3.0, 0.5, 30.0)
  theta TVV(20.0, 2.0, 200.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.05
  sigma PROP ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
";

const IOV_MODEL: &str = r"
[parameters]
  theta TVCL(3.0, 0.5, 30.0)
  theta TVV(20.0, 2.0, 200.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.05
  kappa KAPPA_CL ~ 0.03
  sigma PROP ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
";

/// Twelve subjects, two occasions each, so `α = 0.5` gives a six-subject batch
/// and the κ kernel has occasions to move.
fn population(with_occasions: bool) -> Population {
    let subjects = (0..12usize)
        .map(|i| {
            let f = i as f64;
            Subject {
                id: format!("{}", i + 1),
                doses: vec![
                    DoseEvent::new(0.0, 100.0 + 5.0 * f, 1, 0.0, false, 0.0),
                    DoseEvent::new(24.0, 100.0 + 5.0 * f, 1, 0.0, false, 0.0),
                ],
                dose_occasions: if with_occasions {
                    vec![1, 2]
                } else {
                    Vec::new()
                },
                obs_times: vec![1.0, 6.0, 25.0, 30.0],
                observations: vec![
                    4.5 + 0.3 * f,
                    3.0 + 0.2 * (f % 4.0),
                    5.5 + 0.25 * f,
                    3.6 + 0.1 * (f % 5.0),
                ],
                obs_cmts: vec![1; 4],
                occasions: if with_occasions {
                    vec![1, 1, 2, 2]
                } else {
                    Vec::new()
                },
                cens: vec![0; 4],
                ..Default::default()
            }
        })
        .collect();
    Population {
        subjects,
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

fn opts(batch_fraction: Option<f64>) -> FitOptions {
    let mut o = FitOptions {
        saem_n_exploration: 6,
        saem_n_convergence: 4,
        saem_n_mh_steps: 4,
        saem_omega_burnin: 0,
        saem_seed: Some(2_607),
        run_covariance_step: false,
        verbose: false,
        ..FitOptions::default()
    };
    if let Some(a) = batch_fraction {
        o.saem_batch_fraction = a;
    }
    o
}

fn run_bits(model: &CompiledModel, pop: &Population, o: &FitOptions) -> Vec<u64> {
    let r = run_saem(model, pop, &model.default_params, o).expect("SAEM run");
    assert!(
        r.ofv.is_finite(),
        "SAEM objective must be finite, got {}",
        r.ofv
    );
    let mut out = vec![r.ofv.to_bits()];
    out.extend(r.params.theta.iter().map(|v| v.to_bits()));
    out.extend(r.params.omega.matrix.iter().map(|v| v.to_bits()));
    out.extend(r.params.sigma.values.iter().map(|v| v.to_bits()));
    for e in &r.eta_hats {
        out.extend(e.iter().map(|v| v.to_bits()));
    }
    for subj in &r.kappas {
        for k in subj {
            out.extend(k.iter().map(|v| v.to_bits()));
        }
    }
    out
}

// ─── The batch draw ───────────────────────────────────────────────────────

#[test]
fn full_fraction_draws_no_batch() {
    // `None` is what keeps the default run on the pre-#2607 path: no mask, no
    // extra RNG stream, no per-subject gate taken.
    assert!(saem_batch_mask(40, 1.0, 12345, 1).is_none());
    // A fraction whose ceiling is the whole population is also full batch —
    // 0.99 of 12 subjects is 12 subjects, not 11.
    assert_eq!(batch_size(12, 0.99), 12);
    assert!(saem_batch_mask(12, 0.99, 12345, 7).is_none());
}

#[test]
fn batch_has_the_requested_size_and_is_reproducible() {
    for &(n, a, want) in &[(100, 0.25, 25), (10, 0.1, 1), (7, 0.5, 4), (3, 0.01, 1)] {
        assert_eq!(batch_size(n, a), want, "n={n} a={a}");
        let m = saem_batch_mask(n, a, 99, 3).expect("a strict subset");
        assert_eq!(m.iter().filter(|&&b| b).count(), want, "n={n} a={a}");
        // Same (seed, k) → same subset: the run is reproducible at a fixed seed.
        assert_eq!(saem_batch_mask(n, a, 99, 3), Some(m));
    }
}

#[test]
fn batch_moves_between_iterations_and_reaches_every_subject() {
    let n = 50;
    let a = 0.2;
    let masks: Vec<Vec<bool>> = (1..=60)
        .map(|k| saem_batch_mask(n, a, 4242, k).unwrap())
        .collect();
    // Different iterations draw different subsets — a mask fixed across k would
    // freeze 80% of the chains for the whole fit.
    assert_ne!(masks[0], masks[1]);
    // And a different master seed draws a different subset at the same k.
    assert_ne!(saem_batch_mask(n, a, 4243, 1).unwrap(), masks[0]);
    // Over 60 iterations at α = 0.2 every subject's chain is visited, and each
    // is visited roughly α·K = 12 times (a uniform draw, not a fixed rotation).
    for i in 0..n {
        let hits = masks.iter().filter(|m| m[i]).count();
        assert!(
            (2..=30).contains(&hits),
            "subject {i} drawn {hits} times in 60"
        );
    }
}

#[test]
fn batch_fraction_outside_unit_interval_is_rejected() {
    assert_eq!(validate_batch_fraction(1.0), Ok(1.0));
    assert_eq!(validate_batch_fraction(0.25), Ok(0.25));
    for bad in [0.0, -0.5, 1.5, f64::NAN, f64::INFINITY] {
        let e = validate_batch_fraction(bad).expect_err("must reject");
        assert!(e.contains("batch_fraction") && e.contains("(0, 1]"), "{e}");
    }
}

#[test]
fn batch_fraction_fit_option_parses_and_validates() {
    let mut o = FitOptions::default();
    assert_eq!(
        o.saem_batch_fraction, 1.0,
        "the default must stay full batch"
    );
    apply_fit_option(&mut o, "batch_fraction", "0.25").expect("parses");
    assert_eq!(o.saem_batch_fraction, 0.25);
    apply_fit_option(&mut o, "saem_batch_fraction", "1").expect("parses");
    assert_eq!(o.saem_batch_fraction, 1.0);
    for bad in ["0", "-0.1", "1.01", "nan"] {
        let e = apply_fit_option(&mut o, "batch_fraction", bad).expect_err(bad);
        assert!(e.contains("(0, 1]"), "{bad}: {e}");
        assert_eq!(
            o.saem_batch_fraction, 1.0,
            "a rejected value must not stick"
        );
    }
}

// ─── Whole runs ───────────────────────────────────────────────────────────

/// `batch_fraction = 1` is the full-batch algorithm, bit for bit.
///
/// Measured against `main` (`f8955c24`) when this landed: the same fixture's
/// bits under the pre-#2607 `run_saem` equal the default-options bits here.
/// What this test keeps pinned afterwards is that the explicit `1.0` and the
/// default agree, and that a fraction rounding up to the whole population (0.99
/// of 12) takes the same no-batch path rather than drawing an all-`true` mask
/// through a different branch.
#[test]
fn batch_fraction_one_is_bit_identical_to_full_batch() {
    let model = parse_model_string(MODEL).expect("parses");
    let pop = population(false);
    let base = run_bits(&model, &pop, &opts(None));
    assert_eq!(run_bits(&model, &pop, &opts(Some(1.0))), base);
    assert_eq!(run_bits(&model, &pop, &opts(Some(0.99))), base);
}

/// `α < 1` changes the run (the gate is live), stays finite, and is
/// reproducible at a fixed seed.
#[test]
fn half_batch_changes_the_run_and_is_reproducible() {
    let model = parse_model_string(MODEL).expect("parses");
    let pop = population(false);
    let full = run_bits(&model, &pop, &opts(None));
    let half = run_bits(&model, &pop, &opts(Some(0.5)));
    assert_ne!(half, full, "α = 0.5 must not reduce to the full-batch run");
    assert_eq!(run_bits(&model, &pop, &opts(Some(0.5))), half);
}

/// The κ (IOV) kernel is batched too, and the interval step-scale rule meets
/// subjects that went a whole interval undrawn (α = 1/12 → one subject per
/// iteration, interval 2), whose `0/0` rate it must skip rather than read as
/// 0% acceptance. This is a smoke test of that arm — it pins that the path
/// runs, stays finite and is live, not the skip's numerical effect.
#[test]
fn iov_minibatch_with_interval_adaptation_runs() {
    let model = parse_model_string(IOV_MODEL).expect("parses");
    let pop = population(true);
    let mut o = opts(Some(1.0 / 12.0));
    o.saem_scale_adaptation = ScaleAdaptation::Interval;
    o.saem_adapt_interval = 2;
    let full = {
        let mut f = o.clone();
        f.saem_batch_fraction = 1.0;
        run_bits(&model, &pop, &f)
    };
    let mini = run_bits(&model, &pop, &o);
    assert_ne!(mini, full);
}

#[test]
fn run_saem_rejects_a_programmatic_zero_fraction() {
    let model = parse_model_string(MODEL).expect("parses");
    let pop = population(false);
    let e = run_saem(&model, &pop, &model.default_params, &opts(Some(0.0)))
        .err()
        .expect("α = 0 runs no E-step and must be refused");
    assert!(e.contains("batch_fraction"), "{e}");
}

// ─── Robbins-Monro gain under a mini-batch ────────────────────────────────

/// One η on CL, twenty observations per subject at a 2% residual: the
/// conditional of η is ~30× narrower than Ω, so the starting block step is far
/// too large and the default Robbins-Monro controller has to walk the scale
/// down by orders of magnitude. That walk is what the gain index decides.
const RICH_MODEL: &str = r"
[parameters]
  theta TVCL(3.0, 0.5, 30.0)
  theta TVV(20.0, 2.0, 200.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.0001

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
";

fn rich_population() -> Population {
    let times: Vec<f64> = (1..=20).map(|j| 0.6 * j as f64).collect();
    let subjects = (0..16usize)
        .map(|i| {
            let cl = 3.0 * (0.3 * ((i as f64) - 7.5) / 7.5).exp();
            let v = 20.0;
            Subject {
                id: format!("{}", i + 1),
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times: times.clone(),
                observations: times
                    .iter()
                    .enumerate()
                    // ±~1% deterministic residual, so σ stays away from zero
                    // and the conditional of η has a stable width.
                    .map(|(j, &t)| {
                        let e = 0.01 * (((7 * i + 3 * j) % 5) as f64 - 2.0) / 2.0;
                        100.0 / v * (-cl / v * t).exp() * (1.0 + e)
                    })
                    .collect(),
                obs_cmts: vec![1; times.len()],
                cens: vec![0; times.len()],
                ..Default::default()
            }
        })
        .collect();
    Population {
        subjects,
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

fn tail_acceptance(batch_fraction: f64, k1: usize, k2: usize) -> f64 {
    let model = parse_model_string(RICH_MODEL).expect("parses");
    let o = FitOptions {
        saem_n_exploration: k1,
        saem_n_convergence: k2,
        saem_n_mh_steps: 4,
        saem_omega_burnin: 0,
        saem_seed: Some(2_607),
        // The default controller, without the dead band, so every iteration's
        // rate takes a step and the gain sequence is the only thing compared.
        saem_scale_adaptation: ScaleAdaptation::RobbinsMonro,
        saem_scale_deadband: None,
        saem_batch_fraction: batch_fraction,
        run_covariance_step: false,
        verbose: false,
        ..FitOptions::default()
    };
    let r = run_saem(&model, &rich_population(), &model.default_params, &o).expect("SAEM run");
    r.saem_mh_accept_tail.expect("a post-burn-in tail")
}

/// Same number of moves per chain → the same adaptation (#2607 review).
///
/// The fixture starts the block step ~30× too large (one η, twenty 1%-noise
/// observations per subject, start scale 0.3), so the default Robbins-Monro
/// controller has to walk `log δ` down by ~3 before the chain mixes.
///
/// The full-batch run moves each chain 80 times. The α = 0.1 and α = 0.25
/// runs get 1/α times the iterations, so each chain moves about as often. With
/// the gain indexed by the subject's own move count, every chain walks the
/// same `Σ j^-0.6`, and the mini-batch tail acceptance matches full batch.
/// Indexed by the global iteration `k` instead, the j-th move of a chain gets
/// the gain of iteration ≈ j/α, a factor `α^0.6` smaller. The chain is then
/// still far off target when the tail window is read.
///
/// Measured (macOS arm64) tail acceptance:
///
/// | arm | per-subject index | global `k` |
/// |---|---|---|
/// | full batch, K = 80 | 0.221 | 0.221 |
/// | α = 0.1, K = 800 | 0.274 | 0.071 |
/// | α = 0.25, K = 320 | 0.269 | 0.127 |
///
/// The full-batch tail is the lowest of the three because its 100-iteration
/// window still holds the opening transient. The mini-batch windows sit later
/// in each chain's life.
///
/// Mutation, run: passing `k` instead of `rm_index` to both `rm_scale_update`
/// calls in `run_saem` reddens both mini-batch assertions.
#[test]
fn minibatch_robbins_monro_gain_follows_the_subjects_own_moves() {
    let full = tail_acceptance(1.0, 40, 40);
    assert!(full.is_finite());
    // The fixture must need the controller and the controller must have
    // delivered: from a ~0% start, the full-batch chain mixes by the tail.
    assert!(
        full > 0.15,
        "full-batch tail {full:.4}: controller did not adapt"
    );
    for (alpha, k) in [(0.1, 400), (0.25, 160)] {
        let mini = tail_acceptance(alpha, k, k);
        assert!(mini.is_finite());
        assert!(
            mini > full - 0.05,
            "α = {alpha}: mini-batch tail {mini:.4} vs full-batch {full:.4} at equal moves per \
             chain — the Robbins-Monro gain is not indexed by the subject's own moves"
        );
    }
}

// ─── Frozen pre-#2607 result ──────────────────────────────────────────────

/// FNV-1a over every bit [`run_bits`] collects, for the analytic and the IOV
/// fixture at default options.
fn default_options_fingerprint() -> u64 {
    let analytic = parse_model_string(MODEL).expect("parses");
    let iov = parse_model_string(IOV_MODEL).expect("parses");
    let mut bits = run_bits(&analytic, &population(false), &opts(None));
    bits.extend(run_bits(&iov, &population(true), &opts(None)));
    bits.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, &b| {
        (h ^ b).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The default (`batch_fraction = 1`) run equals the **pre-#2607** `run_saem`
/// bit for bit, against a fingerprint frozen from `main` at `f8955c24`.
///
/// The constants were produced by this exact function compiled against the
/// pre-change `run_saem`, which has no `saem_batch_fraction` field and no batch
/// code at all, so they are an independent oracle rather than a second run of
/// the new path. Linux is the reference platform. The macOS value is pinned too
/// because fits differ there through the OS math library (`AGENTS.md`). On any
/// other platform the test only checks that the fingerprint is stable across
/// two runs.
///
/// A deliberate numerical change to SAEM's default path will redden this test.
/// Regenerate both constants from the new `main` in that PR, and say so there.
#[test]
fn default_run_matches_the_frozen_pre_2607_fingerprint() {
    let got = default_options_fingerprint();
    let frozen: Option<u64> = if cfg!(target_os = "linux") {
        Some(LINUX_FINGERPRINT)
    } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        Some(MACOS_ARM64_FINGERPRINT)
    } else {
        None
    };
    match frozen {
        Some(want) => assert_eq!(
            got, want,
            "default-options SAEM no longer matches the pre-#2607 run: got {got:#018x}"
        ),
        None => assert_eq!(got, default_options_fingerprint()),
    }
}

const LINUX_FINGERPRINT: u64 = 0x0b30_8cc4_b0d3_21ec;
const MACOS_ARM64_FINGERPRINT: u64 = 0x77dd_a54e_d9a4_2e1d;
