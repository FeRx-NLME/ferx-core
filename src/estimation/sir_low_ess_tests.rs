//! #1723: the low-ESS warning, its floor probe, and `sir_scale = natural`.
use super::*;
use std::cell::Cell;

// Each sentence of the warning, as the substring that identifies it. Every test
// below asserts each one present or absent, so deleting any sentence from
// `low_ess_warning` reddens at least one row.
const S_ESS: &str = "effective sample size is 3.5 of 1000 draws";
const S_THRESHOLD: &str =
    "below the 100 at which the proposal is adequate, so these intervals rest on few draws.";
const S_HEAVIEST: &str =
    "heaviest draw carries 52.9% of the weight; it moves ETA_CL by -6.36 proposal standard \
     deviations, to 4.585e-4 on the reported scale.";
const S_MORE: &str = "Increase `sir_samples` for more stable intervals.";
const S_SHELF_HEAD: &str = "The data do not bound ";
const S_SHELF_TAIL: &str = "its SIR lower limit reflects the parameter box rather than the data.";
const S_TO_NATURAL: &str = "`sir_scale = natural` makes these lower limits independent of the box.";
const S_TO_PACKED: &str = "Under `sir_scale = natural` a variance informed by few groups has a \
                           heavy upper tail; `sir_scale = packed` may sample it better.";

fn heaviest() -> HeaviestDraw {
    HeaviestDraw {
        share: 0.529,
        name: "ETA_CL".into(),
        sd_units: -6.357,
        value: 4.585e-4,
    }
}

/// The measured `warfarin_iov` FOCEI floor probe (#1723 §0).
fn wiov_probe() -> Vec<FloorProbe> {
    [
        ("ETA_CL", 9.31),
        ("ETA_V", 18.09),
        ("ETA_KA", 1.56),
        ("KAPPA_CL", 76.78),
    ]
    .into_iter()
    .map(|(n, d)| FloorProbe {
        name: n.into(),
        dofv: d,
        variance: 6.14e-6,
    })
    .collect()
}

fn has(msg: &str, s: &str) -> bool {
    msg.contains(s)
}

/// Row "ESS ≥ 100": nothing is said and the probe never runs; one ULP-scale
/// step below, it is. Both sides of the gate in one test, so moving the
/// threshold to 99.9 (the `>` mutation's twin) or to `<=` reddens it. Also
/// S5: the probe is the expensive half, and the call counter pins that a
/// healthy run spends nothing on it.
#[test]
fn low_ess_warning_gate_straddles_the_threshold_and_skips_the_probe_above_it() {
    let calls = Cell::new(0);
    let probe = || {
        calls.set(calls.get() + 1);
        wiov_probe()
    };
    assert_eq!(
        low_ess_warning(100.0, 1000, SirScale::Packed, &heaviest(), probe),
        None
    );
    assert_eq!(calls.get(), 0, "the probe ran on a healthy run");
    let below = 100.0f64.next_down();
    let probe = || {
        calls.set(calls.get() + 1);
        wiov_probe()
    };
    let msg = low_ess_warning(below, 1000, SirScale::Packed, &heaviest(), probe).expect("warns");
    assert_eq!(calls.get(), 1);
    assert!(has(&msg, "effective sample size is 100.0 of 1000"), "{msg}");
}

/// Row "ESS < 100, no flagged variance": ESS, draw and more-samples sentences;
/// no shelf sentence and **no** claim that anything *is* bounded away from zero
/// (a conditional ΔOFV ≥ 3.84 does not prove the profile is), and no
/// `sir_scale` advice — nothing predicts `natural` would help.
#[test]
fn low_ess_warning_without_a_flagged_variance_names_the_draw_only() {
    let probe = || {
        wiov_probe()
            .into_iter()
            .filter(|p| p.name != "ETA_KA")
            .collect()
    };
    let msg = low_ess_warning(3.548, 1000, SirScale::Packed, &heaviest(), probe).unwrap();
    for s in [S_ESS, S_THRESHOLD, S_HEAVIEST, S_MORE] {
        assert!(has(&msg, s), "missing {s:?} in {msg}");
    }
    for s in [
        S_SHELF_HEAD,
        S_SHELF_TAIL,
        S_TO_NATURAL,
        S_TO_PACKED,
        "sir_scale",
    ] {
        assert!(!has(&msg, s), "unexpected {s:?} in {msg}");
    }
    assert!(!msg.contains("bounded away"), "{msg}");
}

/// Row "ESS < 100, flagged variance": the above **plus** the flagged name with
/// its ΔOFV and variance, the box sentence, and the pointer to `natural` — and
/// only the flagged coordinate is named in the shelf sentence.
#[test]
fn low_ess_warning_names_each_flagged_variance_and_only_those() {
    let msg = low_ess_warning(3.548, 1000, SirScale::Packed, &heaviest(), wiov_probe).unwrap();
    for s in [
        S_ESS,
        S_THRESHOLD,
        S_HEAVIEST,
        S_MORE,
        S_SHELF_HEAD,
        S_SHELF_TAIL,
        S_TO_NATURAL,
        "ETA_KA (ΔOFV 1.56 at variance 6.14e-6) away from zero",
        "raises the OFV by less than χ²₁(0.95) = 3.84",
    ] {
        assert!(has(&msg, s), "missing {s:?} in {msg}");
    }
    assert!(!has(&msg, S_TO_PACKED), "{msg}");
    for unflagged in ["ETA_CL (ΔOFV", "ETA_V (ΔOFV", "KAPPA_CL (ΔOFV"] {
        assert!(!has(&msg, unflagged), "{unflagged} named in {msg}");
    }
}

/// Row "heaviest move is a θ": the draw sentence names the θ, the variances are
/// flagged separately, and nothing says the θ sits near a box.
#[test]
fn low_ess_warning_keeps_a_theta_move_apart_from_the_variance_flags() {
    let h = HeaviestDraw {
        name: "TVE0".into(),
        sd_units: 4.26,
        value: 52.76,
        share: 0.12,
    };
    let msg = low_ess_warning(35.2, 1000, SirScale::Packed, &h, wiov_probe).unwrap();
    assert!(has(&msg, "it moves TVE0 by +4.26 proposal"), "{msg}");
    assert!(has(&msg, "do not bound ETA_KA (ΔOFV"), "{msg}");
    assert!(!has(&msg, "TVE0 (ΔOFV"), "{msg}");
}

/// The χ²₁(0.95) gate from both sides: exactly the quantile is *not* flagged
/// (zero is then on the boundary of the LR interval, not inside it), one step
/// below is. `<` → `<=` reddens the first half; a cut at 3.84 the second.
#[test]
fn low_ess_warning_flags_strictly_below_the_chi2_quantile() {
    let at = |d: f64| {
        let probe = move || {
            vec![FloorProbe {
                name: "ETA_KA".into(),
                dofv: d,
                variance: 6.14e-6,
            }]
        };
        low_ess_warning(3.5, 1000, SirScale::Packed, &heaviest(), probe).unwrap()
    };
    assert!(!has(&at(CHI2_1_95), S_SHELF_HEAD));
    assert!(has(&at(CHI2_1_95.next_down()), S_SHELF_HEAD));
    assert!(has(&at(3.8414), S_SHELF_HEAD), "3.8414 < χ²₁(0.95)");
}

/// Row "every variance FIX": the probe has nothing to move, so no shelf or
/// scale sentence — the ESS and draw part only.
#[test]
fn low_ess_warning_with_nothing_to_probe_is_the_draw_part_only() {
    let msg = low_ess_warning(3.548, 1000, SirScale::Packed, &heaviest(), Vec::new).unwrap();
    for s in [S_ESS, S_THRESHOLD, S_HEAVIEST, S_MORE] {
        assert!(has(&msg, s), "missing {s:?} in {msg}");
    }
    for s in [S_SHELF_HEAD, S_SHELF_TAIL, S_TO_NATURAL, S_TO_PACKED] {
        assert!(!has(&msg, s), "unexpected {s:?} in {msg}");
    }
}

/// Rows "natural": below the threshold, the ESS, draw and more-samples
/// sentences plus the pointer back to `packed`; never the box sentence, and
/// the probe never runs (the natural target is box-independent). At or above
/// the threshold, nothing.
#[test]
fn low_ess_warning_under_natural_points_back_to_packed_and_never_probes() {
    let calls = Cell::new(0);
    let probe = || {
        calls.set(calls.get() + 1);
        wiov_probe()
    };
    let msg = low_ess_warning(3.548, 1000, SirScale::Natural, &heaviest(), probe).unwrap();
    assert_eq!(calls.get(), 0, "the floor probe ran under natural");
    for s in [S_ESS, S_THRESHOLD, S_HEAVIEST, S_MORE, S_TO_PACKED] {
        assert!(has(&msg, s), "missing {s:?} in {msg}");
    }
    for s in [S_SHELF_HEAD, S_SHELF_TAIL, S_TO_NATURAL] {
        assert!(!has(&msg, s), "unexpected {s:?} in {msg}");
    }
    assert_eq!(
        low_ess_warning(100.0, 1000, SirScale::Natural, &heaviest(), wiov_probe),
        None
    );
}

/// The model the probe and the run-level tests share: a free Ω, a FIX Ω, a
/// free κ, a 2×2 `block_omega`, a θ and a σ. Six subjects on two occasions.
const PROBE_MODEL: &str = "
[parameters]
  theta TVCL(5.0, 0.01, 100)
  theta TVV(50.0, FIX)
  theta TVKA(1.5, FIX)
  omega ETA_CL ~ 0.09
  omega ETA_KA ~ 0.1 FIX
  block_omega (ETA_V, ETA_Q) = [0.04, 0.01, 0.05]
  kappa KAPPA_CL ~ 0.02
  sigma PROP_ERR ~ 0.15 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV * exp(ETA_V)
  KA = TVKA * exp(ETA_KA + ETA_Q)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  iov_column = OCC
";

fn probe_fixture() -> (CompiledModel, Population) {
    let model = crate::parser::model_parser::parse_model_string(PROBE_MODEL).expect("parse");
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("d.csv");
    let mut csv = String::from("ID,TIME,AMT,EVID,CMT,DV,MDV,OCC\n");
    for id in 1..=6 {
        for occ in 1..=2 {
            let t0 = (occ - 1) as f64 * 100.0;
            let f = 1.0 + 0.07 * id as f64 - 0.05 * occ as f64;
            csv.push_str(&format!("{id},{t0},100,1,1,.,1,{occ}\n"));
            for (dt, c) in [(1.0, 1.2), (4.0, 1.5), (12.0, 0.6)] {
                csv.push_str(&format!("{id},{},0,0,1,{},0,{occ}\n", t0 + dt, c * f));
            }
        }
    }
    std::fs::write(&data, csv).unwrap();
    let pop = crate::io::datareader::read_nonmem_csv(&data, None, Some("OCC")).unwrap();
    (model, pop)
}

/// S2: the probe moves the free Ω / κ Cholesky diagonals only — never the FIX
/// `ETA_KA`, a θ, the σ, or a `block_omega` off-diagonal.
#[test]
fn floor_probe_coords_are_the_free_variance_diagonals() {
    let (model, _) = probe_fixture();
    let p = &model.default_params;
    let PackedStart { fixed, .. } = pack_with_bounds(p);
    let names = coordinate_names(p);
    let got: Vec<&str> = floor_probe_coords(p, &fixed)
        .into_iter()
        .map(|i| names[i].as_str())
        .collect();
    assert_eq!(got, ["ETA_CL", "ETA_V", "ETA_Q", "KAPPA_CL"], "{names:?}");
}

/// A SIR run on the probe fixture with only 40 draws, so the ESS is below the
/// threshold whatever the proposal, and `adjust_box` applied first.
fn run_probe_fixture(
    scale: SirScale,
    adjust_box: impl FnOnce(&mut PackedBounds),
) -> Result<SirResult, String> {
    let (model, pop) = probe_fixture();
    let params = model.default_params.clone();
    let n = crate::estimation::parameterization::packed_len(&params);
    let PackedStart { fixed, .. } = pack_with_bounds(&params);
    let mut cov = DMatrix::zeros(n, n);
    for i in (0..n).filter(|&i| !fixed[i]) {
        cov[(i, i)] = 0.04;
    }
    let etas = vec![DVector::zeros(4); pop.subjects.len()];
    let opts = FitOptions {
        sir_samples: 40,
        sir_resamples: 20,
        sir_seed: Some(3),
        sir_scale: scale,
        verbose: false,
        ..FitOptions::default()
    };
    let ofv = {
        let (ehs, hms, _, kp) = crate::estimation::inner_optimizer::run_inner_loop_warm(
            &model,
            &pop,
            &params,
            opts.inner_maxiter,
            opts.inner_tol,
            Some(&etas),
            Some(&compute_mu_k(&model, &params.theta, opts.mu_referencing)),
            0,
            0,
        );
        2.0 * pop_nll_opts(&model, &pop, &params, &ehs, &hms, &kp, &opts)
    };
    run_sir_in_box(&model, &pop, &params, &etas, &cov, ofv, &opts, adjust_box)
}

/// S2, run level: the probe reads each coordinate's **own** lower bound. With
/// `ETA_CL`'s floor moved to −3 the variance it reports is `e⁻⁶ = 2.48e-3`;
/// probing at the shipped constant (−6) instead would report `6.14e-6`.
#[test]
fn floor_probe_moves_each_coordinate_to_its_own_box_floor() {
    let (model, _) = probe_fixture();
    let i = coordinate_names(&model.default_params)
        .iter()
        .position(|n| n == "ETA_CL")
        .unwrap();
    let r = run_probe_fixture(SirScale::Packed, |b| b.lower[i] = -3.0).expect("sir");
    let w = r
        .warnings
        .iter()
        .find(|w| w.starts_with("effective sample size"))
        .unwrap_or_else(|| panic!("no low-ESS warning: {:?}", r.warnings));
    assert!(r.effective_sample_size < SIR_LOW_ESS);
    assert!(w.contains("ETA_CL (ΔOFV"), "ETA_CL not flagged: {w}");
    assert!(w.contains("at variance 2.48e-3"), "{w}");
    assert!(!w.contains("ETA_KA"), "the FIX ETA_KA was probed: {w}");
}

/// S9, centre: `ŷ + C ∇` on a 2-D fixture with a known `C`, and the clamp
/// into the draw box on the coordinate whose shift overshoots it.
#[test]
fn natural_centre_is_the_laplace_shift_clamped_into_the_box() {
    let c = DMatrix::from_row_slice(2, 2, &[0.04, 0.01, 0.01, 0.09]);
    let g = DVector::from_vec(vec![2.0, 1.0]);
    let b = PackedBounds {
        lower: vec![-10.0, -10.0, -10.0],
        upper: vec![10.0, 10.0, 0.05],
    };
    let got = natural_centre(&[7.0, -1.0, 0.0], &c, &g, &[0, 2], &b);
    assert_eq!(got[1], -1.0, "a held coordinate moved");
    assert_eq!(got[0], 7.0 + (0.04 * 2.0 + 0.01 * 1.0));
    assert_eq!(got[2], 0.05, "shift 0.11 past the 0.05 upper bound");
}

/// The ESS and an FNV hash of every resampled packed vector (each value to 12
/// significant digits) of a 400-draw packed run on the probe fixture — through
/// `run_sir_core` with default options, i.e. the API and the default that
/// existed before #1723. Rounded so an ULP of a platform's `exp` / `ln` cannot
/// move it; every mutation below moves values in their leading digits.
fn packed_fixture_digest() -> (f64, u64) {
    let (model, pop) = probe_fixture();
    let params = model.default_params.clone();
    let n = crate::estimation::parameterization::packed_len(&params);
    let PackedStart { fixed, .. } = pack_with_bounds(&params);
    let mut cov = DMatrix::zeros(n, n);
    for i in (0..n).filter(|&i| !fixed[i]) {
        cov[(i, i)] = 0.01;
    }
    let etas = vec![DVector::zeros(4); pop.subjects.len()];
    let opts = FitOptions {
        sir_samples: 400,
        sir_resamples: 100,
        sir_seed: Some(11),
        sir_keep_samples: true,
        verbose: false,
        ..FitOptions::default()
    };
    let r = run_sir_core(&model, &pop, &params, &etas, &cov, 0.0, &opts).expect("sir");
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in r.resamples_packed.as_ref().unwrap() {
        for x in v {
            for b in format!("{x:.11e};").bytes() {
                h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
            }
        }
    }
    (r.effective_sample_size, h)
}

/// S9, packed half: the default is the SIR that shipped before #1723. The ESS
/// is `f64::from_bits` of the value this fixture produced on `202bea5e` (macOS
/// arm64), where its resampled vectors were also bit-identical to this
/// branch's; the hash is of those same vectors. Re-centring under `Packed`,
/// adding the natural weight under `Packed`, or flipping the default each
/// move the hash.
#[test]
fn packed_sir_is_unchanged_from_before_1723() {
    assert_eq!(FitOptions::default().sir_scale, SirScale::Packed);
    let (ess, hash) = packed_fixture_digest();
    let want = f64::from_bits(PACKED_ESS_BITS);
    assert!(
        ess.is_finite() && ((ess - want) / want).abs() < 1e-12,
        "ESS {ess}, before #1723 {want}"
    );
    assert_eq!(hash, PACKED_RESAMPLE_HASH, "resampled vectors moved");
}
// Measured on `202bea5e` and on this branch, macOS arm64 and Linux (see the
// PR); the four fit-level fixtures (`warfarin`, `warfarin_iov` FOCEI and FOCE,
// `mbma_placebo`) were compared the same way, ESS / resample-hash / CI bits.
const PACKED_ESS_BITS: u64 = 4_609_556_852_108_965_889;
const PACKED_RESAMPLE_HASH: u64 = 4_345_404_384_744_949_393;

/// S7 (#1723): under `natural` the ω²_KA lower limit does not depend on the
/// box floor; under `packed` it does. `warfarin_iov` FOCEI, where ETA_KA's
/// floor ΔOFV is 1.56, 40 000 draws, the Ω / κ diagonal floors moved from the
/// shipped −6 to −4 and −9 (one variable: the floor), both scales in one test.
///
/// Measured (macOS arm64, 20 000 resamples): packed 5.05e-4 → 1.01e-4,
/// |ln ratio| 1.610 — the limit sits just above the floor's e⁻⁸ = 3.4e-4 at −4;
/// natural 5.04e-3 → 5.30e-3, |ln ratio| 0.050, percentile noise. The bounds
/// (natural < 0.15, packed > 0.8) give 3× and 2× headroom; the packed half
/// asserts the straddle, so a fixture on which the floor stopped mattering
/// cannot pass as box independence.
///
/// Independence alone is not enough, and mutation showed it: with the natural
/// Jacobian deleted (the re-centre kept) the shifted proposal never reaches the
/// shelf, so the limit is 3.89e-4 at **both** floors — "independent" — at ESS
/// 120. So the natural half also pins the level, ω²_KA lower within
/// `|ln(lo / 5.17e-3)| < 0.4` (measured 0.025 at both floors; that mutation
/// 2.59), and the ESS above 4 000 (measured 8 771.5 / 8 776.5; that mutation
/// 120.1 / 123.6).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn natural_scale_lower_limit_is_box_independent_where_packed_is_not() {
    use crate::estimation::uncertainty_samples::fitted_params_from_result;
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let prep = crate::api::prepare_run(
        root.join("examples/warfarin_iov.ferx").to_str().unwrap(),
        Some(root.join("data/warfarin_iov.csv").to_str().unwrap()),
    )
    .unwrap();
    let opts = FitOptions {
        verbose: false,
        method: EstimationMethod::FoceI,
        interaction: true,
        run_covariance_step: true,
        sir: false,
        ..prep.parsed.fit_options.clone()
    };
    let model = &prep.parsed.model;
    let f = crate::api::fit(model, &prep.population, &prep.init_params, &opts).unwrap();
    let params = fitted_params_from_result(&f, model);
    let etas: Vec<DVector<f64>> = f.subjects.iter().map(|s| s.eta.clone()).collect();
    let cov = f.covariance_matrix.clone().expect("covariance");
    let PackedStart { fixed, .. } = pack_with_bounds(&params);
    let floors = floor_probe_coords(&params, &fixed);
    let ka_lower = |scale: SirScale, floor: f64| {
        let o = FitOptions {
            sir_samples: 40_000,
            sir_resamples: 20_000,
            sir_seed: Some(1705),
            sir_scale: scale,
            ..opts.clone()
        };
        let r = run_sir_in_box(
            model,
            &prep.population,
            &params,
            &etas,
            &cov,
            f.ofv,
            &o,
            |b| {
                for &i in &floors {
                    b.lower[i] = floor;
                }
            },
        )
        .expect("sir");
        let lo = r.ci_omega[2].0;
        eprintln!(
            "S7 {scale:?} floor {floor}: ESS {:.1} ω²_KA lower {lo:e}",
            r.effective_sample_size
        );
        assert!(lo.is_finite() && lo > 0.0, "{lo}");
        if scale == SirScale::Natural {
            let ess = r.effective_sample_size;
            assert!(ess > 4_000.0, "natural ESS {ess} at floor {floor}");
            assert!(
                (lo / 5.17e-3).ln().abs() < 0.4,
                "natural ω²_KA lower {lo:e} at floor {floor}"
            );
        }
        lo
    };
    let packed = (ka_lower(SirScale::Packed, -4.0) / ka_lower(SirScale::Packed, -9.0)).ln();
    let natural = (ka_lower(SirScale::Natural, -4.0) / ka_lower(SirScale::Natural, -9.0)).ln();
    eprintln!(
        "S7 |ln ratio| packed {:.3} natural {:.3}",
        packed.abs(),
        natural.abs()
    );
    assert!(
        packed.abs() > 0.8,
        "packed no longer tracks the floor: {packed}"
    );
    assert!(natural.abs() < 0.15, "natural tracks the floor: {natural}");
}
