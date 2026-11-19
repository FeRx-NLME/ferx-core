use super::*;
use std::collections::HashMap;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

// ---------------------------------------------------------------------------
// Tier 1: the accumulator
// ---------------------------------------------------------------------------

/// Deterministic standard normals (Box–Muller over a 64-bit LCG): the test needs draws it
/// can reproduce, not a statistical RNG.
struct Normals(u64);
impl Normals {
    fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn z(&mut self) -> f64 {
        (-2.0 * self.u().ln()).sqrt() * (2.0 * std::f64::consts::PI * self.u()).cos()
    }
}

/// The accumulated value is `−2 Σ_i log mean_t exp(−nll_{i,t})`, written out naively on values
/// small enough for a raw exponential. Each subject has its own maximum arriving at a
/// different step, so the rescale-on-new-maximum branch and the plain-add branch both run.
#[test]
fn accumulator_is_minus_two_log_of_the_mean_density() {
    let draws: [[f64; 3]; 2] = [[3.0, 1.0, 2.5], [0.5, 4.0, 4.5]];
    let mut acc = SaemObjAccumulator::new(2, true, false);
    for t in 0..3 {
        acc.update(&[draws[0][t], draws[1][t]]);
    }
    let want: f64 =
        draws.iter().map(|d| -2.0 * (d.iter().map(|v| (-v).exp()).sum::<f64>() / 3.0).ln()).sum();
    let got = acc.value().unwrap();
    assert!((got - want).abs() < 1e-12, "{got} vs {want}");
}

/// Nothing is ever exponentiated raw: at NLLs of 1e4 `exp(−nll)` underflows to zero, and the
/// value is still the exact log-mean-exp.
/// #2986: with `n_chains` the NLL slice holds every chain's copy of every subject
/// (`chain · N + subject`). The chains pool into their subject's mean: one update with two
/// chains is exactly two single-chain updates with the same draws — not a sum over 2N subjects.
#[test]
fn accumulator_pools_chain_copies_into_their_subject() {
    let (a1, b1, a2, b2) = (3.1, 7.4, 2.6, 8.2);
    let mut pooled = SaemObjAccumulator::new(2, true, false);
    pooled.update(&[a1, b1, a2, b2]);
    let mut serial = SaemObjAccumulator::new(2, true, false);
    serial.update(&[a1, b1]);
    serial.update(&[a2, b2]);
    assert_eq!(pooled.value(), serial.value());
    // No random effect: every chain's value is the same; the first is read.
    let mut last = SaemObjAccumulator::new(2, true, true);
    last.update(&[a1, b1, a1, b1]);
    let mut last1 = SaemObjAccumulator::new(2, true, true);
    last1.update(&[a1, b1]);
    assert_eq!(last.value(), last1.value());
}

#[test]
fn accumulator_survives_nlls_whose_exponential_underflows() {
    let mut acc = SaemObjAccumulator::new(1, true, false);
    for _ in 0..4 {
        acc.update(&[10_000.0]);
    }
    acc.update(&[10_000.0 + 2.0_f64.ln()]);
    // mean of e^{-1e4} (x4) and e^{-1e4}/2: exp(-1e4)·(4 + 0.5)/5
    let want = 2.0 * 10_000.0 - 2.0 * (4.5_f64 / 5.0).ln();
    assert!((acc.value().unwrap() - want).abs() < 1e-9);
}

/// No draw, a disabled accumulator (mixture / IOV / no η), and a subject whose draws all
/// diverged (the 1e20 sentinel) report no value rather than a number.
#[test]
fn accumulator_reports_nothing_it_cannot_stand_behind() {
    assert!(SaemObjAccumulator::new(2, true, false).value().is_none());
    let mut off = SaemObjAccumulator::new(2, false, false);
    off.update(&[1.0, 2.0]);
    assert!(off.value().is_none());
    let mut bad = SaemObjAccumulator::new(2, true, false);
    bad.update(&[1.0, 1e20]);
    assert!(bad.value().is_none());
}

/// `n_η = 0`: the objective is the data `−2 log L` at the end, the last draw's value, not a
/// mean over iterations at moving parameters.
#[test]
fn without_random_effects_the_objective_is_the_last_value() {
    let mut acc = SaemObjAccumulator::new(2, true, true);
    acc.update(&[5.0, 7.0]);
    acc.update(&[1.0, 2.0]);
    assert!((acc.value().unwrap() - 2.0 * 3.0).abs() < 1e-12);
}

/// The convention itself, on the linear-Gaussian model where it is closed-form. One η per
/// subject, `y_j = η + ε`, `η ~ N(0, ω)`, `ε ~ N(0, s²)`: at a posterior draw `η ~ N(m, v)`,
/// `−log p(y, η) = c + (η − m)²/(2v)` with `c` its minimum, so `E[−2 log p] = 2c + 1` while the
/// mean inside the log gives `−2 log(e^{−c}/√2) = 2c + ln 2`: the accumulated value sits
/// `n_η (ln 2 − 1) = −0.307` below the posterior expectation, the offset NONMEM's SAEMOBJ shows.
/// The `nll` is written out here, not taken from the library.
#[test]
fn linear_gaussian_offset_from_the_posterior_expectation_is_ln2_minus_one() {
    let (omega, s2) = (0.5_f64, 0.25_f64);
    let subjects: [&[f64]; 3] = [&[0.4, 0.9, 0.2], &[-1.0, -0.2], &[0.1, 0.3, 0.6, 0.2]];
    let nll = |y: &[f64], eta: f64| {
        0.5 * (eta * eta / omega
            + omega.ln()
            + y.iter().map(|v| s2.ln() + (v - eta).powi(2) / s2).sum::<f64>())
    };
    let mut rng = Normals(7);
    let k = 200_000;
    let mut acc = SaemObjAccumulator::new(subjects.len(), true, false);
    let mut expectation = vec![0.0; subjects.len()];
    let posts: Vec<(f64, f64)> = subjects
        .iter()
        .map(|y| {
            let v = 1.0 / (1.0 / omega + y.len() as f64 / s2);
            (v * y.iter().sum::<f64>() / s2, v)
        })
        .collect();
    for _ in 0..k {
        let nlls: Vec<f64> =
            subjects.iter().zip(&posts).map(|(y, (m, v))| nll(y, m + v.sqrt() * rng.z())).collect();
        for (e, n) in expectation.iter_mut().zip(&nlls) {
            *e += 2.0 * n / k as f64;
        }
        acc.update(&nlls);
    }
    let c_sum: f64 = subjects.iter().zip(&posts).map(|(y, (m, _))| nll(y, *m)).sum();
    let got = acc.value().unwrap();
    let exact = 2.0 * c_sum + subjects.len() as f64 * 2.0_f64.ln();
    let expect_sum: f64 = expectation.iter().sum();
    // Monte-Carlo error of the mean of e^{-z²/2} at k draws: ≈ 0.39/√k per subject in the log.
    assert!((got - exact).abs() < 0.02, "{got} vs closed form {exact}");
    let offset = (got - expect_sum) / subjects.len() as f64;
    assert!(
        (offset - (2.0_f64.ln() - 1.0)).abs() < 0.01,
        "offset {offset} vs ln2-1 = {}",
        2.0_f64.ln() - 1.0
    );
}

// ---------------------------------------------------------------------------
// Tier 2: complete ferx SAEM fits against NONMEM's printed SAEMOBJ
// ---------------------------------------------------------------------------

fn ext_first_table_final(name: &str) -> HashMap<String, f64> {
    let text = std::fs::read_to_string(root().join("nonmem_anchor/results").join(name)).unwrap();
    let mut names: Vec<String> = Vec::new();
    let mut tables = 0;
    let mut last = HashMap::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.first() {
            None => continue,
            Some(&"TABLE") => tables += 1,
            Some(&"ITERATION") => names = f.iter().map(|s| s.to_string()).collect(),
            Some(&"-1000000000") if tables == 1 => {
                last = names
                    .iter()
                    .cloned()
                    .zip(f.iter().map(|v| v.parse::<f64>().unwrap()))
                    .collect();
            }
            Some(_) => {}
        }
    }
    assert!(!last.is_empty(), "{name}: no final row in the SAEM table");
    last
}

/// (ferx model, data, NONMEM .ext): the 1-η additive-mu-reference anchor (6 subjects) and the
/// two-η covariate anchors, the fast ones.
const ANCHORS: &[(&str, &str, &str)] = &[
    ("saem_additive_mu.ferx", "nonmem_anchor/saem_additive_mu.csv", "saem_additive_mu.ext"),
    ("covmuref_additive_saem_fit.ferx", "data/covmuref_additive.csv", "covmuref_additive_saem.ext"),
    ("covmuref_power_saem_fit.ferx", "data/covmuref_power.csv", "covmuref_power_saem.ext"),
];

/// Worst `|ferx − NONMEM| / (N·n_η)` of a complete ferx SAEM fit (its own estimates, its own
/// chain) against NONMEM's own run, measured on Linux (debug): see the `eprintln!` of every
/// anchor. Both sides are one Monte-Carlo run, and the estimates differ a little.
const ANCHOR_BAND_PER_ETA: f64 = 0.04;

#[test]
fn saem_objective_of_a_ferx_fit_matches_nonmems_printed_saemobj() {
    let mut worst = 0.0_f64;
    for (model, data, ext) in ANCHORS {
        let nm = ext_first_table_final(ext)["SAEMOBJ"];
        let prep = crate::prepare_run(
            root().join("nonmem_anchor").join(model).to_str().unwrap(),
            Some(root().join(data).to_str().unwrap()),
        )
        .expect("prepare");
        let opts = crate::types::FitOptions {
            verbose: false,
            run_covariance_step: false,
            ..prep.parsed.fit_options.clone()
        };
        let r = crate::fit(&prep.parsed.model, &prep.population, &prep.init_params, &opts)
            .expect("fit");
        let obj = r.saem_objective.as_ref().expect("a SAEM objective");
        assert_eq!(obj.kind, crate::types::SaemOfv::Saem, "{ext}");
        let n_ne = (prep.population.subjects.len() * prep.parsed.model.n_eta) as f64;
        let off = (r.ofv - nm) / n_ne;
        assert!(off.is_finite(), "{ext}");
        eprintln!(
            "ANCHOR {ext}: NONMEM SAEMOBJ {nm:.4}  ferx {:.4}  N*neta {n_ne}  (ferx-NONMEM)/(N*neta) {off:+.5}",
            r.ofv
        );
        worst = worst.max(off.abs());
        assert!(off.abs() <= ANCHOR_BAND_PER_ETA, "{ext}: {off:+.5} per N*neta");
    }
    assert!(worst.is_finite());
}
