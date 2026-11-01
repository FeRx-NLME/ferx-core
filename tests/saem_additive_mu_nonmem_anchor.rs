//! NONMEM anchor for SAEM's additive mu-reference (#2540).
//!
//! `nonmem_anchor/saem_additive_mu.ferx` is `level_se_mbma_fit.ferx` with every θ but
//! TVE0 held at NONMEM `level_se_a`'s optimum, so only TVE0 and ω² are free and η
//! enters linearly with a fixed residual: FOCEI is the exact likelihood, and
//! `level_se_a`'s objective (165.826023) is the MLE of this model too.
//!
//! - `saem_additive_mu.ctl`: NONMEM SAEM with `MU_1 = THETA(1)`, `E0 = MU_1 + ETA(1)`,
//!   then `METHOD=1 MAXEVAL=0` to score the estimate. It reaches the MLE: 165.826056
//!   at seed 12345, and 165.8260–165.8261 over seeds 12345, 2, 3, 7, 11.
//! - `saem_additive_mu_nomu.ctl`: the same without `MU_1`. 171.417778 at seed 12345,
//!   167.41–174.32 over the five seeds. The un-mu-referenced shape is off in NONMEM
//!   too, which is what makes this a defect of the shape and not of ferx's sampler.
//!
//! ferx before #2540 took no additive mu-reference at all, so `E0 = TVE0 + ETA_E0`
//! landed where NONMEM's no-`MU` run does (168.54–172.76 over the same five seeds).
//! With the natural-scale shift it reaches 165.8260–165.8266. TVE0's lower bound is
//! 0, so it packs as `log θ`: this fixture runs the repacking arm of the shift, the
//! one the shipped MBMA example needs.
//!
//! The pair straddles the gate: the inline spelling (`BASE = TVE0 + PLAFIX +
//! ETA_E0`, `saem_additive_mu_inline.ferx`) is not mu-referenced before or after
//! the fix, so it must stay off the MLE while the split one reaches it.

use std::collections::HashMap;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The final row (`ITERATION = -1000000000`) of the **last** table in a NONMEM `.ext`,
/// by column name. In these streams that is the `MAXEVAL=0` FOCEI step, whose `OBJ`
/// is the exact −2 log L at the SAEM estimate.
fn ext_last(name: &str) -> HashMap<String, f64> {
    let text = std::fs::read_to_string(root().join("nonmem_anchor/results").join(name)).unwrap();
    let mut names: Vec<String> = Vec::new();
    let mut last = HashMap::new();
    for line in text.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        match fields.first() {
            Some(&"TABLE") | None => continue,
            Some(&"ITERATION") => names = fields.iter().map(|s| s.to_string()).collect(),
            Some(&"-1000000000") => {
                last = names
                    .iter()
                    .cloned()
                    .zip(fields.iter().map(|v| v.parse::<f64>().unwrap()))
                    .collect();
            }
            Some(_) => {}
        }
    }
    assert!(!last.is_empty(), "{name}: no final estimates");
    last
}

fn fit(model: &str) -> ferx_core::types::FitResult {
    let prep = ferx_core::prepare_run(
        root().join("nonmem_anchor").join(model).to_str().unwrap(),
        Some(
            root()
                .join("nonmem_anchor/saem_additive_mu.csv")
                .to_str()
                .unwrap(),
        ),
    )
    .expect("prepare");
    let opts = ferx_core::types::FitOptions {
        verbose: false,
        ..prep.parsed.fit_options.clone()
    };
    ferx_core::fit(
        &prep.parsed.model,
        &prep.population,
        &prep.init_params,
        &opts,
    )
    .expect("fit")
}

#[test]
fn saem_additive_mu_reference_reaches_the_mle_like_nonmem_mu() {
    let mle = ext_last("level_se_a.ext")["OBJ"];
    let nm_mu = ext_last("saem_additive_mu.ext");
    let nm_nomu = ext_last("saem_additive_mu_nomu.ext");
    // The oracle itself: NONMEM with MU reaches the MLE, NONMEM without it does not.
    assert!(
        (nm_mu["OBJ"] - mle).abs() < 1e-3,
        "NONMEM MU {} vs MLE {mle}",
        nm_mu["OBJ"]
    );
    assert!(
        nm_nomu["OBJ"] > mle + 1.0,
        "NONMEM no-MU {} vs MLE {mle}",
        nm_nomu["OBJ"]
    );

    let split = fit("saem_additive_mu.ferx");
    let inline = fit("saem_additive_mu_inline.ferx");
    assert!(split.ofv.is_finite() && inline.ofv.is_finite());
    // The realised gaps, so a passing run on another platform still records its digits.
    eprintln!(
        "saem_additive_mu: split OFV - MLE {:.3e}, TVE0 - NONMEM {:.3e}, omega2 - NONMEM \
         {:.3e}, inline OFV - MLE {:.3}",
        split.ofv - mle,
        split.theta[0] - nm_mu["THETA1"],
        split.omega[(0, 0)] - nm_mu["OMEGA(1,1)"],
        inline.ofv - mle
    );

    // Measured: 1.571e-4 above the MLE at the default seed, the same printed digits on
    // macOS arm64 and Linux x86_64 (slow-tests run 41222935329), and at most 6.3e-4
    // above it over seeds 12345, 2, 3, 7, 11 (macOS). The bound is 8× that spread, so a platform
    // whose chain takes a different path still passes, while the pre-#2540 behaviour
    // (≥ 2.7 above on every seed) fails by three orders of magnitude.
    assert!(
        (split.ofv - mle).abs() < 5e-3,
        "split E0 = TVE0 + ETA_E0: OFV {} vs MLE {mle}",
        split.ofv
    );
    // TVE0 48.767 against NONMEM 48.758 (five-seed ferx 48.759–48.781); ω² 5.174
    // against 5.164 (ferx 5.135–5.210). Bounds about twice the worst five-seed gap.
    assert!(
        (split.theta[0] - nm_mu["THETA1"]).abs() < 0.05,
        "TVE0: ferx {} vs NONMEM MU {}",
        split.theta[0],
        nm_mu["THETA1"]
    );
    assert!(
        (split.omega[(0, 0)] - nm_mu["OMEGA(1,1)"]).abs() < 0.15,
        "omega2: ferx {} vs NONMEM MU {}",
        split.omega[(0, 0)],
        nm_mu["OMEGA(1,1)"]
    );

    // The control: inline, TVE0 is not mu-referenced and stays on the ridge (measured
    // 172.663 at the default seed, 168.54–172.76 over five).
    assert!(
        inline.ofv > mle + 1.0,
        "inline BASE = TVE0 + PLAFIX + ETA_E0 must stay off the MLE: {}",
        inline.ofv
    );

    // What each spelling is told. The inline one names BASE and the split that fixes
    // it; the split one is not warned at all.
    let not_mu = |f: &ferx_core::types::FitResult| {
        f.warnings
            .iter()
            .find(|w| w.contains("not mu-referenced:"))
            .cloned()
    };
    let w = not_mu(&inline).expect("inline: non-mu-referenced warning");
    assert!(w.contains("not mu-referenced: BASE"), "{w}");
    assert!(w.contains("`E0 = TVE0 + ETA_E0`"), "{w}");
    assert!(w.contains("`BASE = E0 + ...`"), "{w}");
    assert!(w.contains("summed with other terms in one line"), "{w}");
    assert_eq!(not_mu(&split), None, "split: {:?}", split.warnings);
}
