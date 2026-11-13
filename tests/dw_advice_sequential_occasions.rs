//! #2858: what the positive Durbin–Watson warning advises on a compartment-free
//! model whose occasions follow one another in time, and that the advice is true.
//!
//! Since #2829 the statistic is taken across such occasions. The warning used to
//! fall back to the no-grouping text ("a `correlation ar1(RHO) by COL` line, with
//! COL any data column that labels the group"), and the only such column the model
//! knows is the occasion column. That line is the one that cannot model the pairs
//! the statistic fired on: a block never joins rows of two occasions. The warning
//! now advises a line that spans the subject, `by ID`.
//!
//! The statistic under a `correlation` line was blind in the same way: it paired
//! within blocks only, so under `by OCC` it never took the cross-occasion pairs,
//! and with one row per occasion it had none. Blocks that run in time are now one
//! series, as occasions are, and the warning under such a line names `by ID` too.
//!
//! Every claim of the new text is run as a fit here, on
//! `tests/data/dw_sequential_algebraic/` (24 subjects on one clock, a line in time
//! with a subject intercept, AR(1) residuals with ρ = 0.7 across the whole subject;
//! `sparse.csv` has one row per occasion, `rich.csv` three). The ungated tests are
//! evaluations at the initial estimates (`outer_maxiter = 0`), which is enough for
//! each claim: the statistic and the warning are post-fit arithmetic, and a line
//! that enters the likelihood moves the objective at any ρ ≠ 0. One gated test
//! converges. There is no NONMEM column: NONMEM prints no Durbin–Watson statistic
//! and no advice.
//!
//! ## Measured (macOS arm64, `9733956f` + this change)
//!
//! Linux x86_64 (slow-tests run 43406353558 on this branch) prints the same digits
//! for every OFV, ρ̂ and statistic below that it reports.
//!
//! At the initial estimates, ρ = 0.3, model with a kappa at `OCC`:
//!
//! | data | line | OFV | DW | Durbin–Watson warning |
//! |---|---|---|---|---|
//! | sparse | none | −13.844880 | 0.9364 | the new advice |
//! | sparse | `ar1(RHO) by OCC` | −13.844880 (10 ULP away) | 0.9364 (was `NaN`) | "remains …", names `by ID` (was none) |
//! | sparse | `ar1(RHO) by ID` | −50.043571 | 1.2346 | "remains … decorrelated by" |
//! | rich | none | −105.896587 | 1.3318 | the new advice |
//! | rich | `ar1(RHO) by OCC` | −142.124218 | 1.6785 (was 1.8384) | none |
//! | rich | `ar1(RHO) by ID` | −154.451388 | 1.7904 | none |
//!
//! Converged, sparse: no line −37.739613 (DW 0.9675, warns); `by OCC` −37.739613,
//! ρ left at 0.3, DW 0.9675 and the warning (was `NaN`, none); `by ID`
//! −124.232414, ρ̂ = 0.8152, DW 2.1623, silent.
//! Converged, rich: no line −206.449257 (DW 2.3365, the kappa absorbs it); `by OCC`
//! −211.834593, DW 2.1751, silent; `by ID` −246.651176.
//!
//! "Was" is the statistic split at every residual block, before this change joined
//! blocks that run in time. No objective in these tables moved with it.

use ferx_core::api::read_population_for;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::{classify_warning, WarningCode};
use ferx_core::{fit, FitOptions, FitResult, Population};

const SPARSE: &str = "tests/data/dw_sequential_algebraic/sparse.csv";
const RICH: &str = "tests/data/dw_sequential_algebraic/rich.csv";

/// How the model declares its occasions. All three put a kappa on the intercept.
#[derive(Clone, Copy, Debug)]
enum Grouping {
    /// `iov_column = OCC`.
    IovColumn,
    /// One `level OCC by OCC in ID`.
    SingleLevel,
    /// `level PER by PER in ID`, `level OCC by OCC in PER`: nested, both in time.
    Nested,
    /// `iov_column = OCC` with no kappa: the occasions are declared and nothing
    /// absorbs what they share.
    IovColumnNoKappa,
}

const GROUPINGS: [Grouping; 3] = [Grouping::IovColumn, Grouping::SingleLevel, Grouping::Nested];

/// The compartment-free model under `grouping`, with `line` in `[error_model]`
/// (`""` for none; a line reads the θ `RHO`, declared at 0.3 when there is one).
fn model(grouping: Grouping, line: &str) -> String {
    let (levels, fit_options) = match grouping {
        Grouping::IovColumn => ("  kappa K_E0 ~ 0.04\n", "  iov_column = OCC\n"),
        Grouping::IovColumnNoKappa => ("", "  iov_column = OCC\n"),
        Grouping::SingleLevel => ("  level OCC by OCC in ID\n  kappa K_E0 ~ 0.04 at OCC\n", ""),
        Grouping::Nested => (
            "  level PER by PER in ID\n  level OCC by OCC in PER\n  kappa K_E0 ~ 0.04 at OCC\n",
            "",
        ),
    };
    let kappa = if levels.contains("kappa") {
        " + K_E0"
    } else {
        ""
    };
    let (rho, line) = if line.is_empty() {
        (String::new(), String::new())
    } else {
        (
            "  theta RHO(0.3, 0.0, 0.95)\n".to_string(),
            format!("  {line}\n"),
        )
    };
    format!(
        "model seq\n[parameters]\n  theta TVE0(10.0, 0.1, 100.0)\n  theta TVSL(0.5, -10.0, 10.0)\n\
         {rho}  omega ETA_E0 ~ 1.0\n{levels}  sigma ADD_ERR ~ 0.5 (sd)\n\
         [individual_parameters]\n  E0 = TVE0 + ETA_E0{kappa}\n  SL = TVSL\n\
         [structural_model]\n  y = E0 + SL * TIME\n\
         [error_model]\n  DV ~ additive(ADD_ERR)\n{line}\
         [fit_options]\n  method = focei\n  covariance = false\n{fit_options}"
    )
}

fn run(src: &str, data: &str, outer_maxiter: usize) -> (Population, FitResult) {
    let parsed = parse_full_model(src).unwrap_or_else(|e| panic!("model parses: {e}\n{src}"));
    assert!(
        parsed.model.is_algebraic(),
        "the fixture is compartment-free"
    );
    let (pop, _) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        data,
        None,
        parsed.fit_options.iov_column.as_deref(),
        None,
        &parsed.column_map,
    )
    .expect("data loads");
    let opts = FitOptions {
        outer_maxiter,
        run_covariance_step: false,
        ..parsed.fit_options.clone()
    };
    let r = fit(&parsed.model, &pop, &parsed.model.default_params, &opts)
        .unwrap_or_else(|e| panic!("the model must fit: {e}\n{src}"));
    assert!(r.ofv.is_finite(), "OFV {}", r.ofv);
    assert!(
        r.subjects
            .iter()
            .all(|s| s.iwres.iter().all(|e| e.is_finite())),
        "every IWRES must be finite"
    );
    (pop, r)
}

fn evaluate(src: &str, data: &str) -> FitResult {
    run(src, data, 0).1
}

fn dw_warnings(r: &FitResult) -> Vec<&String> {
    r.warnings
        .iter()
        .filter(|w| classify_warning(w).category == WarningCode::DwAutocorrelation)
        .collect()
}

/// An inert line leaves the objective where it was. Not to the bit: the line
/// routes the subject through the dense-`R` path, and the two paths differ by 10
/// ULP here (measured 1.8e-14 on −13.84488). A live line moves it by 36 or more.
fn assert_same_objective(with: f64, without: f64, label: &str) {
    assert!(with.is_finite() && without.is_finite(), "{label}");
    assert!(
        (with - without).abs() < 1e-10,
        "{label}: {with} vs {without}"
    );
}

/// The one Durbin–Watson warning of `r`.
fn the_dw_warning<'a>(r: &'a FitResult, label: &str) -> &'a str {
    let dw = dw_warnings(r);
    assert_eq!(dw.len(), 1, "{label}: {:#?}", r.warnings);
    dw[0]
}

/// Lag pairs and pair-normalised DW of each subject's IWRES as one series, written
/// out here so the statistic is not graded with its own kernel.
fn whole_subject_dw(r: &FitResult) -> (usize, f64) {
    let (mut pairs, mut num, mut den) = (0usize, 0.0, 0.0);
    for s in &r.subjects {
        for w in s.iwres.windows(2) {
            pairs += 1;
            num += (w[1] - w[0]).powi(2);
            den += (w[0] * w[0] + w[1] * w[1]) / 2.0;
        }
    }
    (pairs, num / den)
}

/// Whether every subject has more than one occasion label, each one contiguous run
/// starting strictly later than the row before it (the #2829 rule, recomputed from
/// the public fields).
fn occasions_follow_in_time(pop: &Population) -> bool {
    pop.subjects.iter().all(|s| {
        let mut closed: Vec<u32> = Vec::new();
        s.occasions.windows(2).any(|w| w[0] != w[1])
            && (1..s.occasions.len()).all(|j| {
                if s.occasions[j] == s.occasions[j - 1] {
                    return true;
                }
                closed.push(s.occasions[j - 1]);
                s.obs_times[j] > s.obs_times[j - 1] && !closed.contains(&s.occasions[j])
            })
    })
}

/// The repro of #2858, fixed. A compartment-free model with occasions in time and
/// positively autocorrelated residuals: the statistic is the whole-subject one
/// (the first sentence's "the statistic pairs rows of different occasions"), and
/// the warning advises a line `by ID` — not the occasion column, not a placeholder
/// column, and not declaring a grouping the model already has. With one row per
/// occasion and with several, under each way of declaring the occasions.
///
/// Mutation that must redden this: the `AlgebraicInTime` arm of `dw_advice`
/// dropped (the no-grouping text is back: "by COL", "kappa … at").
#[test]
fn occasions_in_time_are_advised_a_line_by_the_subject() {
    for (data, rows, pairs) in [(SPARSE, 192, 168), (RICH, 288, 264)] {
        for g in GROUPINGS {
            let label = format!("{data} {g:?}");
            let (pop, r) = run(&model(g, ""), data, 0);
            // A nested model's occasions are its level cells, which the fit writes
            // on its own copy; the population read here carries none to check.
            // The pair count below shows the series was not split either way.
            if !matches!(g, Grouping::Nested) {
                assert!(occasions_follow_in_time(&pop), "{label}");
            }
            assert_eq!(
                pop.subjects
                    .iter()
                    .map(|s| s.observations.len())
                    .sum::<usize>(),
                rows,
                "{label}"
            );
            let (n, want) = whole_subject_dw(&r);
            assert_eq!(n, pairs, "{label}: 24 subjects, every consecutive pair");
            assert!(r.dw_statistic.is_finite(), "{label}");
            assert!((r.dw_statistic - want).abs() < 1e-12, "{label}: {want}");
            assert!(r.dw_statistic < 1.5, "{label}: DW {}", r.dw_statistic);
            let msg = the_dw_warning(&r, &label);
            eprintln!(
                "{label}: OFV {:.6} DW {:.6}\n  {msg}",
                r.ofv, r.dw_statistic
            );
            for must in [
                "Positive IWRES autocorrelation detected",
                "across its occasions too",
                "`correlation ar1(RHO) by ID` line",
                "`car1(RHO)`",
                "method = foce or focei",
                "A line `by` the occasions themselves cannot model this",
            ] {
                assert!(msg.contains(must), "{label}: missing `{must}`: {msg}");
            }
            for banned in [
                "by OCC",
                "by PER",
                "by COL",
                "kappa",
                "`level`",
                "labels the group",
            ] {
                assert!(!msg.contains(banned), "{label}: `{banned}` in {msg}");
            }
        }
    }
}

/// "A line `by` the occasions themselves cannot model this: a block never joins
/// rows of two occasions", with one row per occasion. This is where the old advice
/// led. The line `by OCC` leaves the objective of the model without it (to 10 ULP)
/// and `W_RESIDUAL_CORRELATION_INERT` says so.
///
/// The statistic used to be `NaN` here and the Durbin–Watson warning gone, because
/// it paired within blocks only. The blocks run in time, so it is now taken across
/// them: `R` is diagonal, the decorrelated residuals are the IWRES, and the
/// statistic is the one of the model without the line. The warning fires, says the
/// statistic paired rows of consecutive blocks, and names the line `by ID`.
///
/// Mutations that must redden this: the block gate dropped from
/// `split_by_residual_group` (`NaN`, no warning); `dw_across_blocks` always `None`
/// (the warning loses its last two sentences).
#[test]
fn a_line_by_the_occasions_is_inert_and_the_warning_names_the_subject_line() {
    for g in [Grouping::IovColumn, Grouping::SingleLevel] {
        let without = evaluate(&model(g, ""), SPARSE);
        for kind in ["ar1", "car1"] {
            let label = format!("{g:?} {kind}");
            let with = evaluate(
                &model(g, &format!("correlation {kind}(RHO) by OCC")),
                SPARSE,
            );
            eprintln!(
                "sparse {label} by OCC: OFV {:.6} (no line {:.6}) DW {}",
                with.ofv, without.ofv, with.dw_statistic
            );
            assert_same_objective(with.ofv, without.ofv, &label);
            assert!(
                with.dw_statistic.is_finite(),
                "{label}: {}",
                with.dw_statistic
            );
            // Measured: the two agree in every printed digit (0.9363753830495355).
            assert!(
                (with.dw_statistic - without.dw_statistic).abs() < 1e-9,
                "{label}: {} vs {}",
                with.dw_statistic,
                without.dw_statistic
            );
            let msg = the_dw_warning(&with, &label);
            for must in [
                format!("decorrelated by `correlation {kind}(RHO) by OCC`"),
                "The blocks of this line follow one another in time".to_string(),
                "the statistic also pairs rows of consecutive blocks".to_string(),
                "A line `by ID` correlates all rows of one subject.".to_string(),
            ] {
                assert!(msg.contains(&must), "{label}: missing `{must}`: {msg}");
            }
            assert!(!msg.contains("by COL"), "{label}: {msg}");
            assert!(
                with.warnings
                    .iter()
                    .any(|w| w.contains("no block holds more than one observation")),
                "{label}: {:#?}",
                with.warnings
            );
        }
        assert!(
            without.dw_statistic < 1.5,
            "{g:?}: the model without it warns"
        );
    }
}

/// The advised line, as written in the warning, on the data that fired it: it
/// fits, it enters the likelihood (the objective drops at ρ = 0.3 with nothing
/// else moved) and the statistic stays live. Its blocks are whole subjects, so the
/// statistic keeps every pair it had, and at ρ = 0.3 against a true 0.7 the
/// warning still fires, now naming the line.
///
/// Measured (macOS arm64): OFV −13.844880 → −50.043571, DW 0.9364 → 1.2346, for
/// `ar1` and `car1` alike (the visits are one time unit apart, where they are the
/// same matrix).
///
/// Mutation that must redden this: `by ID` in the advice replaced with any
/// spelling that is not the line fitted here (the first assertion reads the line
/// out of the warning).
#[test]
fn the_advised_line_fits_and_keeps_the_warning_live() {
    for g in GROUPINGS {
        let without = evaluate(&model(g, ""), SPARSE);
        let advice = the_dw_warning(&without, &format!("{g:?}"));
        let (pairs, _) = whole_subject_dw(&without);
        for kind in ["ar1", "car1"] {
            let label = format!("{g:?} {kind}");
            let line = format!("correlation {kind}(RHO) by ID");
            // The advice spells `ar1` in full and `car1` as the alternative
            // structure of the same line.
            assert!(
                advice.contains("`correlation ar1(RHO) by ID` line")
                    && advice.contains("`car1(RHO)`"),
                "{label}: the fitted line must be the advised one: {advice}"
            );
            let with = evaluate(&model(g, &line), SPARSE);
            eprintln!(
                "sparse {label} by ID: OFV {:.6} (no line {:.6}) DW {:.6}",
                with.ofv, without.ofv, with.dw_statistic
            );
            assert!(
                with.ofv < without.ofv - 10.0,
                "{label}: the line must enter the likelihood: {} vs {}",
                with.ofv,
                without.ofv
            );
            assert!(with.dw_statistic.is_finite(), "{label}");
            assert!(with.dw_statistic < 1.5, "{label}: DW {}", with.dw_statistic);
            assert_eq!(
                with.subjects
                    .iter()
                    .map(|s| s.iwres.len() - 1)
                    .sum::<usize>(),
                pairs,
                "{label}: one block per subject keeps every pair"
            );
            let msg = the_dw_warning(&with, &label);
            assert!(
                msg.contains(&format!("decorrelated by `{line}`")),
                "{label}: {msg}"
            );
            assert!(
                !with
                    .warnings
                    .iter()
                    .any(|w| w.contains("no block holds more than one observation")),
                "{label}: {:#?}",
                with.warnings
            );
        }
    }
}

/// The same two lines with several rows per occasion, where `by OCC` is not inert:
/// it models the pairs inside an occasion and none across. Both lines evaluated at
/// the same parameters, the line `by ID` reaches the lower objective.
///
/// Measured (macOS arm64) at ρ = 0.3: no line −105.896587; `by OCC` −142.124218;
/// `by ID` −154.451388. The statistic under `by OCC` is taken across its blocks,
/// which run in time: 1.678494, where the split at the blocks gave 1.838370.
///
/// The pooled statistic does not warn: 1.68 is inside the band. Three of a subject's
/// eleven pairs cross an occasion, and it averages them with the eight the line
/// models. The statistic over the 72 pairs that join two blocks is 1.17, and that
/// is what speaks: the cross-block warning names the line `by ID`. Under `by ID`
/// there is one block per subject, no such pair, and no such warning.
///
/// Mutations that must redden this: the block gate dropped (the statistic under
/// `by OCC` reads 1.838370, and no pair joins two blocks); the cross-block warning
/// not pushed by the fit.
#[test]
fn with_several_rows_per_occasion_the_subject_line_reaches_further() {
    for g in [Grouping::IovColumn, Grouping::SingleLevel] {
        let without = evaluate(&model(g, ""), RICH);
        let by_occ = evaluate(&model(g, "correlation ar1(RHO) by OCC"), RICH);
        let by_id = evaluate(&model(g, "correlation ar1(RHO) by ID"), RICH);
        eprintln!(
            "rich {g:?}: no line {:.6}, by OCC {:.6} (DW {:.6}), by ID {:.6} (DW {:.6})",
            without.ofv, by_occ.ofv, by_occ.dw_statistic, by_id.ofv, by_id.dw_statistic
        );
        assert!(
            by_occ.ofv < without.ofv - 10.0,
            "{g:?}: `by OCC` is live here"
        );
        assert!(
            by_id.ofv < by_occ.ofv - 5.0,
            "{g:?}: {} vs {}",
            by_id.ofv,
            by_occ.ofv
        );
        assert!(by_occ.dw_statistic.is_finite() && by_id.dw_statistic.is_finite());
        // 0.16 separates the two splits; the bound leaves the platform 0.02.
        assert!(
            (by_occ.dw_statistic - 1.678494).abs() < 0.02,
            "{g:?}: taken across the blocks, 1.678494; split at them, 1.838370; got {}",
            by_occ.dw_statistic
        );
        let msg = the_dw_warning(&by_occ, &format!("rich {g:?} by OCC"));
        for must in [
            "between consecutive blocks of `correlation ar1(RHO) by OCC`",
            "over the 72 pairs that join two blocks",
            "1.68 over all pairs",
            "A line `by ID` correlates all rows of one subject.",
        ] {
            assert!(msg.contains(must), "{g:?}: missing `{must}`: {msg}");
        }
        // Measured: 1.17 across the blocks, 0.33 under the threshold.
        assert!(dw_warnings(&by_id).is_empty(), "{:#?}", by_id.warnings);
    }
}

/// A text `ID`: `by ID` cannot be read (a `by` column is a number), so that line
/// is inert, and the warning asks for a numeric subject-level column instead. The
/// column it describes, `UNIT`, fits and enters the likelihood.
///
/// Mutation that must redden this: `by_id` always `true` in `dw_advice` (the
/// warning names `by ID` on data where that line couples nothing).
#[test]
fn a_text_id_is_advised_a_numeric_subject_column() {
    let text = std::fs::read_to_string(SPARSE).expect("fixture");
    let rows: Vec<String> = text
        .lines()
        .enumerate()
        .map(|(i, l)| {
            if i == 0 {
                l.to_string()
            } else {
                format!("S{l}")
            }
        })
        .collect();
    let dir = std::env::temp_dir().join(format!(
        "ferx_dw_advice_text_id_{}_{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("sparse_text_id.csv");
    std::fs::write(&path, rows.join("\n") + "\n").expect("temp file");
    let data = path.to_str().expect("utf-8 path");

    let g = Grouping::IovColumn;
    let without = evaluate(&model(g, ""), data);
    let msg = the_dw_warning(&without, "text id");
    eprintln!("text id: {msg}");
    for must in [
        "`correlation ar1(RHO) by COL` line",
        "a numeric data column that has one value per subject",
        "`ID` is not numeric in this data",
    ] {
        assert!(msg.contains(must), "missing `{must}`: {msg}");
    }
    assert!(!msg.contains("by ID"), "{msg}");

    let by_id = evaluate(&model(g, "correlation ar1(RHO) by ID"), data);
    assert_same_objective(by_id.ofv, without.ofv, "`by ID` on a text id is inert");
    assert!(
        by_id
            .warnings
            .iter()
            .any(|w| w.contains("no block holds more than one observation")),
        "{:#?}",
        by_id.warnings
    );
    let by_unit = evaluate(&model(g, "correlation ar1(RHO) by UNIT"), data);
    eprintln!(
        "text id: no line {:.6}, by UNIT {:.6}",
        without.ofv, by_unit.ofv
    );
    assert!(
        by_unit.ofv < without.ofv - 10.0,
        "{} vs {}",
        by_unit.ofv,
        without.ofv
    );
    assert!(by_unit.dw_statistic.is_finite());
    let _ = std::fs::remove_dir_all(&dir);
}

/// The advised line run to convergence on the sparse data: ρ is estimated, the
/// objective drops far below the model without the line, and the statistic ends
/// inside the band with no warning. The line `by OCC` converges to the objective of
/// the model without it, ρ at its initial value.
///
/// Measured (macOS arm64, `9733956f` + this change; not a Linux number): no line
/// −37.739613, DW 0.9675; `by OCC` −37.739613, ρ 0.3; `by ID` −124.232414,
/// ρ̂ 0.8152 (the data were drawn at 0.7), DW 2.1623. The bounds below leave the
/// estimate ±0.15 and the drop 36 of its 86 points.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn the_advised_line_converges_and_resolves_the_warning() {
    let g = Grouping::IovColumn;
    let (_, without) = run(&model(g, ""), SPARSE, 200);
    let (_, by_occ) = run(&model(g, "correlation ar1(RHO) by OCC"), SPARSE, 200);
    let (_, by_id) = run(&model(g, "correlation ar1(RHO) by ID"), SPARSE, 200);
    let rho = |r: &FitResult| r.theta[2];
    eprintln!(
        "converged: no line {:.6} (DW {:.6}); by OCC {:.6} rho {:.6} (DW {}); by ID {:.6} rho \
         {:.6} (DW {:.6})",
        without.ofv,
        without.dw_statistic,
        by_occ.ofv,
        rho(&by_occ),
        by_occ.dw_statistic,
        by_id.ofv,
        rho(&by_id),
        by_id.dw_statistic
    );
    assert!(without.converged && by_id.converged);
    assert_eq!(dw_warnings(&without).len(), 1, "{:#?}", without.warnings);
    assert!((by_occ.ofv - without.ofv).abs() < 1e-4, "{}", by_occ.ofv);
    assert_eq!(rho(&by_occ), 0.3, "an inert ρ stays where it started");
    // Inert, so its decorrelated residuals are the IWRES of the model without it:
    // the same statistic (measured 0.9675346 both), and the warning names `by ID`.
    assert!(by_occ.dw_statistic.is_finite());
    assert!(
        (by_occ.dw_statistic - without.dw_statistic).abs() < 1e-4,
        "{} vs {}",
        by_occ.dw_statistic,
        without.dw_statistic
    );
    let msg = the_dw_warning(&by_occ, "converged by OCC");
    assert!(
        msg.contains("A line `by ID` correlates all rows of one subject."),
        "{msg}"
    );
    assert!(
        by_id.ofv < without.ofv - 50.0,
        "{} vs {}",
        by_id.ofv,
        without.ofv
    );
    assert!((rho(&by_id) - 0.8152).abs() < 0.15, "rho {}", rho(&by_id));
    assert!(
        (1.5..=2.5).contains(&by_id.dw_statistic),
        "DW {}",
        by_id.dw_statistic
    );
    assert!(dw_warnings(&by_id).is_empty(), "{:#?}", by_id.warnings);

    // Three rows per occasion: `by OCC` is live, and `by ID` still reaches further
    // (measured −211.834593 against −246.651176, a gap of 34.8). The pooled statistic
    // under `by OCC` is 2.175118, inside the band; over the 72 cross-block pairs it is
    // 1.476, so the cross-block warning fires here too. That is 0.024 under the
    // threshold, so it is printed, not asserted.
    let (_, rich_occ) = run(&model(g, "correlation ar1(RHO) by OCC"), RICH, 200);
    let (_, rich_id) = run(&model(g, "correlation ar1(RHO) by ID"), RICH, 200);
    eprintln!(
        "converged, rich: by OCC {:.6} rho {:.6} DW {:.6} {:?}; by ID {:.6} rho {:.6}",
        rich_occ.ofv,
        rho(&rich_occ),
        rich_occ.dw_statistic,
        dw_warnings(&rich_occ),
        rich_id.ofv,
        rho(&rich_id)
    );
    assert!(rich_occ.converged && rich_id.converged);
    assert!(
        rich_id.ofv < rich_occ.ofv - 15.0,
        "{} vs {}",
        rich_id.ofv,
        rich_occ.ofv
    );
    assert!(dw_warnings(&rich_id).is_empty(), "{:#?}", rich_id.warnings);

    // Without a kappa nothing absorbs what an occasion shares, and the margin is
    // wide: measured OFV −210.671097, pooled 1.938758, across blocks 1.3579.
    let g = Grouping::IovColumnNoKappa;
    let (_, plain) = run(&model(g, "correlation ar1(RHO) by OCC"), RICH, 200);
    eprintln!(
        "converged, rich, no kappa: by OCC {:.6} DW {:.6} {:?}",
        plain.ofv,
        plain.dw_statistic,
        dw_warnings(&plain)
    );
    assert!(plain.converged);
    assert!(
        (1.5..=2.5).contains(&plain.dw_statistic),
        "the pooled statistic is inside the band: {}",
        plain.dw_statistic
    );
    let msg = the_dw_warning(&plain, "converged, rich, no kappa");
    assert!(
        msg.contains("between consecutive blocks of `correlation ar1(RHO) by OCC`")
            && msg.contains("over the 72 pairs that join two blocks")
            && msg.contains("A line `by ID` correlates all rows of one subject."),
        "{msg}"
    );
}

/// A copy of `sparse.csv` with its `ID` header respelt, in a directory of this
/// test's own.
fn sparse_with_id_header(test: &str, header: &str) -> (std::path::PathBuf, String) {
    let text = std::fs::read_to_string(SPARSE).expect("fixture");
    let text = text.replacen("ID,", &format!("{header},"), 1);
    assert!(
        text.starts_with(&format!("{header},TIME")),
        "the header row"
    );
    let dir = std::env::temp_dir().join(format!("ferx_dw_advice_{test}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("sparse.csv");
    std::fs::write(&path, text).expect("temp file");
    let data = path.to_str().expect("utf-8 path").to_string();
    (dir, data)
}

/// `by` is case-sensitive and the reader's `ID` role is not. On a file whose
/// subject column is spelt `id`, the line `by ID` is refused, so the advice names
/// the column as the file spells it, and that line fits.
///
/// Mutation that must redden this: `SubjectLine::of` naming `ID` whatever the
/// header (the advice then names a line this file refuses).
#[test]
fn the_advice_names_the_subject_column_as_the_data_spell_it() {
    let (dir, data) = sparse_with_id_header("lower_case_id", "id");
    let g = Grouping::IovColumn;
    let without = evaluate(&model(g, ""), &data);
    let msg = the_dw_warning(&without, "header `id`");
    assert!(
        msg.contains("`correlation ar1(RHO) by id` line") && !msg.contains("by ID"),
        "{msg}"
    );
    let by_id = evaluate(&model(g, "correlation ar1(RHO) by id"), &data);
    eprintln!(
        "header `id`: no line {:.6}, by id {:.6}",
        without.ofv, by_id.ofv
    );
    assert!(
        by_id.ofv < without.ofv - 10.0,
        "{} vs {}",
        by_id.ofv,
        without.ofv
    );

    // The straddle: the spelling the advice must not use is refused on this file.
    let parsed = parse_full_model(&model(g, "correlation ar1(RHO) by ID")).expect("parses");
    let (pop, _) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        &data,
        None,
        parsed.fit_options.iov_column.as_deref(),
        None,
        &parsed.column_map,
    )
    .expect("data loads");
    let opts = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        ..parsed.fit_options.clone()
    };
    let err = fit(&parsed.model, &pop, &parsed.model.default_params, &opts)
        .err()
        .expect("`by ID` must be refused on a file whose header is `id`");
    assert!(err.to_string().contains("not found in data"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A `[data] ID = SUBJ` mapping renames the subject column to `ID`, so the advice
/// says `by ID` and that line reads the mapped column: the same objective as on
/// the file with an `ID` header (measured −50.043571 both).
#[test]
fn a_mapped_subject_column_is_read_by_id() {
    let (dir, data) = sparse_with_id_header("mapped_id", "SUBJ");
    let g = Grouping::IovColumn;
    let mapped = |line: &str| {
        format!(
            "model seq\n[data]\n  path = {data}\n  ID = SUBJ\n{}",
            model(g, line).replacen("model seq\n", "", 1)
        )
    };
    let without = evaluate(&mapped(""), &data);
    let msg = the_dw_warning(&without, "mapped");
    assert!(
        msg.contains("`correlation ar1(RHO) by ID` line") && !msg.contains("SUBJ"),
        "{msg}"
    );
    let by_id = evaluate(&mapped("correlation ar1(RHO) by ID"), &data);
    let reference = evaluate(&model(g, "correlation ar1(RHO) by ID"), SPARSE);
    eprintln!(
        "mapped ID = SUBJ: by ID {:.6}, on the ID file {:.6}",
        by_id.ofv, reference.ofv
    );
    assert!(
        by_id.ofv < without.ofv - 10.0,
        "{} vs {}",
        by_id.ofv,
        without.ofv
    );
    assert!(
        (by_id.ofv - reference.ofv).abs() < 1e-9,
        "{} vs {}",
        by_id.ofv,
        reference.ofv
    );
    let _ = std::fs::remove_dir_all(&dir);
}
