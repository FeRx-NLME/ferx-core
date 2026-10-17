//! Simulating a model with nested levels (#1082 PR4): one κ per group of its own level.
//!
//! ## The degenerate oracles
//!
//! A nested model whose κ all live at **one** level draws exactly what a one-level IOV
//! model draws when its occasion column labels that level's groups: the per-level draw
//! visits the same groups in the same order and consumes the same normals. So
//!
//! - κ only at `STUDY` (the `ARM` level carries none) equals `iov_column = STUDYID`, and
//! - κ only at `ARM` equals `iov_column = ARMID`,
//!
//! bit for bit, on every simulated row, with one seed. The two sides run different code:
//! the nested side groups by level through `NestedLayout::draw` on the cells
//! `assign_level_cells` writes at the simulate chokepoint; the one-level side draws one κ
//! per occasion group as `simulate()` always has. Each oracle is the test of its own
//! level's draw: a study κ drawn per arm fails the first (two arms of a study read two
//! draws), an arm κ drawn per study fails the second.
//!
//! ## One level is today's simulation
//!
//! [`one_level_simulation_is_unchanged`] pins the seeded rows of a one-level IOV model
//! to the bits `main` produced before #1082 PR4 (`2685ec5e`), so moving the one-level
//! draw into `KappaDraw` changed nothing.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{simulate_with_seed, CompiledModel, Population, SimOutcome, Subject};
use std::collections::HashMap;

/// `(STUDY, ARM)` per observation row, for one subject. Unbalanced on purpose: study 1
/// has three arms, study 2 one, study 3 two, and arm labels repeat across studies (nested
/// by key). Two rows per arm, so each arm's rows must share its κ too.
const ROWS: [(f64, f64); 12] = [
    (1.0, 1.0),
    (1.0, 1.0),
    (1.0, 2.0),
    (1.0, 2.0),
    (1.0, 3.0),
    (1.0, 3.0),
    (2.0, 1.0),
    (2.0, 1.0),
    (3.0, 1.0),
    (3.0, 1.0),
    (3.0, 2.0),
    (3.0, 2.0),
];

fn snap(subject: usize, study: f64, arm: f64) -> HashMap<String, f64> {
    [
        ("STUDY".to_string(), study),
        ("ARM".to_string(), arm),
        ("STUDYID".to_string(), 10.0 * subject as f64 + study),
        (
            "ARMID".to_string(),
            100.0 * subject as f64 + 10.0 * study + arm,
        ),
    ]
    .into_iter()
    .collect()
}

/// Three subjects of [`ROWS`]. With `occasion_column`, the occasion labels a one-level
/// model reads are that column's values (what `iov_column` would read); a nested model's
/// labels are left empty, for the simulate chokepoint to assign.
fn population(occasion_column: Option<&str>) -> Population {
    let subjects = (1..=3)
        .map(|i| {
            let covs: Vec<HashMap<String, f64>> =
                ROWS.iter().map(|&(s, a)| snap(i, s, a)).collect();
            let occasions = occasion_column
                .map(|c| covs.iter().map(|cv| cv[c] as u32).collect())
                .unwrap_or_default();
            Subject {
                id: i.to_string(),
                obs_times: (0..ROWS.len()).map(|j| j as f64 + 1.0).collect(),
                observations: vec![f64::NAN; ROWS.len()],
                obs_cmts: vec![1; ROWS.len()],
                cens: vec![0; ROWS.len()],
                covariates: covs[0].clone(),
                obs_covariates: covs,
                occasions,
                ..Default::default()
            }
        })
        .collect();
    Population {
        subjects,
        covariate_names: ["STUDY", "ARM", "STUDYID", "ARMID"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// `B = TVB·exp(ETA) + K`: the row's κ enters additively, so IPRED − TVB·exp(η) is the
/// κ the row read. `kappa_line` declares `K` (with or without `at`), `levels` the level
/// lines, `tail` any `[fit_options]`.
fn model(levels: &str, kappa_line: &str, tail: &str) -> CompiledModel {
    parse_model_string(&format!(
        "[parameters]\n  theta TVB(10.0)\n  omega ETA ~ 0.09\n{levels}  {kappa_line}\n  \
         sigma ADD ~ 0.25\n[individual_parameters]\n  B = TVB * exp(ETA) + K\n\
         [structural_model]\n  y = B\n[error_model]\n  DV ~ additive(ADD)\n{tail}"
    ))
    .expect("fixture parses")
}

const LEVELS: &str = "  level STUDY by STUDY in ID\n  level ARM by ARM in STUDY\n";

/// Every simulated row as bits: `(id, time, ipred, value)`.
fn rows_bits(m: &CompiledModel, p: &Population, seed: u64) -> Vec<(String, u64, u64, u64)> {
    simulate_with_seed(m, p, &m.default_params, 3, seed)
        .expect("simulates")
        .into_iter()
        .map(|r| {
            let SimOutcome::Continuous { value } = r.outcome else {
                panic!("Gaussian rows only")
            };
            (
                format!("{}#{}", r.id, r.sim),
                r.time.to_bits(),
                r.ipred.to_bits(),
                value.to_bits(),
            )
        })
        .collect()
}

fn assert_bit_identical(level: &str, nested: &CompiledModel, one_level: &CompiledModel, col: &str) {
    for seed in [1, 1082] {
        let a = rows_bits(nested, &population(None), seed);
        let b = rows_bits(one_level, &population(Some(col)), seed);
        assert_eq!(a.len(), 3 * 3 * ROWS.len());
        assert_eq!(
            a, b,
            "level {level}: the nested draw (κ at {level} only) must equal one-level IOV \
             by `{col}`, bit for bit (seed {seed})"
        );
    }
}

#[test]
fn a_kappa_at_study_only_draws_as_iov_by_study() {
    let nested = model(LEVELS, "kappa K ~ 4.0 at STUDY", "");
    let one_level = model(
        "",
        "kappa K ~ 4.0",
        "[fit_options]\n  iov_column = STUDYID\n",
    );
    assert_bit_identical("STUDY", &nested, &one_level, "STUDYID");
}

#[test]
fn a_kappa_at_arm_only_draws_as_iov_by_arm() {
    let nested = model(LEVELS, "kappa K ~ 4.0 at ARM", "");
    let one_level = model("", "kappa K ~ 4.0", "[fit_options]\n  iov_column = ARMID\n");
    assert_bit_identical("ARM", &nested, &one_level, "ARMID");
}

/// The oracles above would agree if neither side drew any κ. With κ at both levels the
/// rows of one arm share their κ, two arms of one study differ (the arm κ), and the
/// spread is the sum of both levels' — so the κ is drawn, and drawn per level.
#[test]
fn both_levels_draw_and_the_rows_of_an_arm_share_them() {
    let m = parse_model_string(
        "[parameters]\n  theta TVB(10.0)\n  omega ETA ~ 0.09\n  level STUDY by STUDY in ID\n  \
         level ARM by ARM in STUDY\n  kappa K_ST ~ 4.0 at STUDY\n  kappa K_ARM ~ 1.0 at ARM\n  \
         sigma ADD ~ 0.25\n[individual_parameters]\n  B = TVB * exp(ETA) + K_ST + K_ARM\n\
         [structural_model]\n  y = B\n[error_model]\n  DV ~ additive(ADD)\n",
    )
    .unwrap();
    let rows = simulate_with_seed(&m, &population(None), &m.default_params, 1, 5).unwrap();
    let subject_1: Vec<f64> = rows
        .iter()
        .filter(|r| r.id == "1")
        .map(|r| r.ipred)
        .collect();
    assert_eq!(subject_1.len(), ROWS.len());
    for pair in subject_1.chunks(2) {
        assert!(pair[0].is_finite());
        assert_eq!(
            pair[0], pair[1],
            "the two rows of one arm read one κ: {subject_1:?}"
        );
    }
    assert_ne!(
        subject_1[0], subject_1[2],
        "two arms of study 1 differ: {subject_1:?}"
    );
}

/// Seeded one-level IOV rows (`iov_column = ARMID`, `kappa K ~ 4.0`, seeds 1 and 1082,
/// three replicates) folded into one FNV-1a hash of their bits. The constant is what
/// `main` at `2685ec5e` — before #1082 PR4 — returns for this test.
#[test]
fn one_level_simulation_is_unchanged() {
    let m = model("", "kappa K ~ 4.0", "[fit_options]\n  iov_column = ARMID\n");
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |x: u64| {
        for b in x.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    };
    for seed in [1, 1082] {
        for (_, t, ipred, value) in rows_bits(&m, &population(Some("ARMID")), seed) {
            eat(t);
            eat(ipred);
            eat(value);
        }
    }
    assert_eq!(
        h, ONE_LEVEL_HASH,
        "one-level seeded rows moved: hash {h:#018x}"
    );
}

const ONE_LEVEL_HASH: u64 = 0x2d97_0f91_8913_b10f;

/// A declared zero κ draws exactly 0 under nested levels too (#1605's factor, which the
/// per-level draw restricts by level; review r2 on #1082 PR4). With `K ~ 0.0 FIX at
/// STUDY` every row of a subject reads `TVB·exp(η)` exactly; drawing from the
/// regularised Cholesky factor instead would add ~1e-4 per study. The NPDE reference
/// agrees bit for bit with the one-level spelling, which #1605 already pins.
#[test]
fn a_zero_kappa_at_a_level_draws_exactly_zero() {
    let nested = model(LEVELS, "kappa K ~ 0.0 FIX at STUDY", "");
    let one_level = model(
        "",
        "kappa K ~ 0.0 FIX",
        "[fit_options]\n  iov_column = STUDYID\n",
    );
    let rows = simulate_with_seed(&nested, &population(None), &nested.default_params, 1, 3)
        .expect("simulates");
    for id in ["1", "2", "3"] {
        let ipred: Vec<f64> = rows
            .iter()
            .filter(|r| r.id == id)
            .map(|r| r.ipred)
            .collect();
        assert_eq!(ipred.len(), ROWS.len());
        assert!(ipred[0].is_finite());
        assert!(
            ipred.iter().all(|&v| v.to_bits() == ipred[0].to_bits()),
            "subject {id}: a zero κ at STUDY moved the rows: {ipred:?}"
        );
    }
    assert_eq!(
        npde_bits(&nested, population(None)),
        npde_bits(&one_level, population(Some("STUDYID"))),
        "the NPDE reference draws a zero κ at STUDY as exactly 0"
    );
}

// ── the NPDE reference distribution ──────────────────────────────────────────────

/// The NPDE reference simulates κ through the same per-level draw (`KappaDraw`), so the
/// two oracles above hold for it too: npd and npde bit for bit, per level.
fn npde_bits(m: &CompiledModel, mut p: Population) -> Vec<(u64, u64)> {
    for s in &mut p.subjects {
        s.observations = (0..ROWS.len()).map(|j| 10.0 + 0.3 * j as f64).collect();
    }
    let (scores, _) = ferx_core::stats::npde::compute_npde_npd_diag(
        m,
        &p,
        &m.default_params,
        200,
        Some(11),
        None,
    )
    .expect("npde");
    scores
        .iter()
        .flat_map(|s| s.npd.iter().zip(&s.npde))
        .map(|(a, b)| {
            assert!(a.is_finite(), "npd {a}");
            (a.to_bits(), b.to_bits())
        })
        .collect()
}

#[test]
fn the_npde_reference_draws_a_study_kappa_as_iov_by_study() {
    let nested = model(LEVELS, "kappa K ~ 4.0 at STUDY", "");
    let one_level = model(
        "",
        "kappa K ~ 4.0",
        "[fit_options]\n  iov_column = STUDYID\n",
    );
    assert_eq!(
        npde_bits(&nested, population(None)),
        npde_bits(&one_level, population(Some("STUDYID"))),
        "level STUDY: the NPDE reference must draw one κ per study"
    );
}

#[test]
fn the_npde_reference_draws_an_arm_kappa_as_iov_by_arm() {
    let nested = model(LEVELS, "kappa K ~ 4.0 at ARM", "");
    let one_level = model("", "kappa K ~ 4.0", "[fit_options]\n  iov_column = ARMID\n");
    assert_eq!(
        npde_bits(&nested, population(None)),
        npde_bits(&one_level, population(Some("ARMID"))),
        "level ARM: the NPDE reference must draw one κ per arm"
    );
}
