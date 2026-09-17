//! `W_CMT_DEFAULTED` fires exactly where the compartment choice is observable (#1009).
//!
//! Two rounds of review on PR #1404 found the same failure mode in the suppression
//! predicate: it was written as a **list** of the model classes someone had thought
//! of, and each round the list was missing the class on the next engine. Round 1
//! missed the observation-side dispatchers, round 2 missed an `[odes]` model's own
//! per-CMT readout — and chasing that turned up a third gap, the analytical dose
//! channel, which the predicate had been suppressing wholesale.
//!
//! A third round of "add the arm you just noticed" would be the same mistake, so
//! this file pins the property the arms are *for*, over every analytical topology
//! the engine has:
//!
//! > For each `PkModel`, a dose written into compartment `k` produces a different
//! > prediction from the same dose written into compartment 1 **if and only if**
//! > dropping the `CMT` column from that dataset raises `W_CMT_DEFAULTED`.
//!
//! Neither side is asserted against a hard-coded table. The left side is *measured*
//! by predicting both ways; the right side is read out of `ferx check`; the test
//! asserts they agree. So the expectation cannot drift from the engine, and a
//! topology whose channel map changes reddens this file rather than silently
//! widening or narrowing the warning.
//!
//! The `match` in [`case_for`] is exhaustive over `PkModel` on purpose: a new
//! analytical model is a **compile error** here until someone says which
//! compartments it can be dosed into, which is precisely the step both review
//! rounds skipped.
//!
//! Tier 2: every call returns immediately — `predict` and `validate_model_file`,
//! never `fit`.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{predict, read_nonmem_csv, validate_model_file};
use ferx_core::{CompiledModel, PkModel, Population};
use std::io::Write;
use tempfile::NamedTempFile;

/// Observations far enough apart to see both the absorption and the elimination
/// phase, so a mis-placed dose cannot coincide with the right answer by landing on
/// a crossing point.
const OBS_TIMES: [u32; 3] = [1, 4, 8];

/// Relative gap above which two prediction vectors count as distinguishable.
///
/// This is deliberately **not** a tolerance on an approximation — it separates two
/// kinds of outcome, and the test prints the realised numbers on every run so the
/// separation stays visible rather than assumed. The degenerate side is pinned
/// exactly by [`a_topology_with_no_dose_channels_either_rejects_cmt_or_ignores_it_bit_for_bit`]
/// (bit-identical, not merely close), so the only way this bound could matter is if
/// a *live* dose channel produced a difference below it — which
/// [`the_warning_fires_exactly_where_the_dose_compartment_is_observable`] would
/// report as a disagreement with the predicate rather than absorb.
///
/// Realised separations, printed by that test and recorded here so a future change
/// that narrows one is visible in the diff. The smallest live separation is
/// **3.406e-1** (`OneCptOral`, CMT=2 against CMT=1) — five orders of magnitude above
/// this bound — and every suppressed topology sits at exactly **0.000e0**, because
/// its second compartment is *rejected* rather than silently re-routed. Nothing is
/// anywhere near the bound in either direction:
///
/// | model | probes | rejected | worst rel diff | observable | warned |
/// |---|---|---|---|---|---|
/// | `OneCptIv` | 2 | 2 | 0.000e0 | false | false |
/// | `OneCptOral` | 2 | — | 3.406e-1 | true | true |
/// | `TwoCptIv` | 2 | — | 9.601e-1 | true | true |
/// | `TwoCptOral` | 2, 3 | — | 9.410e-1 | true | true |
/// | `ThreeCptIv` | 2, 3 | — | 9.909e-1 | true | true |
/// | `ThreeCptOral` | 2, 3, 4 | — | 9.867e-1 | true | true |
/// | `OneCptTransit` | 2 | 2 | 0.000e0 | false | false |
/// | `TwoCptTransit` | 2, 3 | 2, 3 | 0.000e0 | false | false |
/// | `OneCptIg` | 2 | 2 | 0.000e0 | false | false |
/// | `TwoCptIg` | 2, 3 | 2, 3 | 0.000e0 | false | false |
const DISTINGUISHABLE: f64 = 1e-6;

struct Case {
    /// The `.ferx` source for this topology.
    src: String,
    /// 1-based compartments to probe besides compartment 1. Every one of them must
    /// be *parseable* as a dose target; whether it is *distinguishable* from
    /// compartment 1 is what the test measures rather than assumes.
    probe_cmts: Vec<usize>,
}

/// A minimal valid model of the given shape.
///
/// `omega ... ~ 0.0 FIX` rather than the bare `~ 0.0` the neighbouring anchor
/// fixtures use: those only ever call `predict`, while this file runs the model
/// through `validate_model_file`, which rejects an unfixed zero variance with
/// `E_OMEGA_INIT_AT_RAIL` (#1229). Fixing it keeps the predictions identical — η is
/// zero either way — and makes the model something `ferx check` will accept, which
/// is the whole point of asking it about the warning.
fn model_src(structural: &str, params: &str, indiv: &str) -> String {
    format!(
        r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
{params}
  omega ETA_CL ~ 0.0 FIX
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
{indiv}

[structural_model]
  {structural}

[error_model]
  DV ~ proportional(PROP)
"#
    )
}

const KA_P: &str = "  theta TVKA(1.0, 0.01, 10.0)";
const KA_I: &str = "  KA = TVKA";
const TWO_P: &str = "  theta TVQ(3.0, 0.1, 50.0)\n  theta TVV2(80.0, 5.0, 500.0)";
const TWO_I: &str = "  Q  = TVQ\n  V2 = TVV2";
const THREE_P: &str = "  theta TVQ(3.0, 0.1, 50.0)\n  theta TVV2(80.0, 5.0, 500.0)\n  \
                       theta TVQ3(1.0, 0.1, 50.0)\n  theta TVV3(120.0, 5.0, 900.0)";
const THREE_I: &str = "  Q  = TVQ\n  V2 = TVV2\n  Q3 = TVQ3\n  V3 = TVV3";
const TRANSIT_P: &str = "  theta TVMTT(2.0, 0.1, 20.0)\n  theta TVNN(3.0, 1.0, 20.0)";
const TRANSIT_I: &str = "  MTT = TVMTT\n  NN = TVNN";
const IG_P: &str = "  theta TVMAT(2.0, 0.1, 20.0)\n  theta TVCV2(0.5, 0.05, 5.0)";
const IG_I: &str = "  MAT = TVMAT\n  CV2 = TVCV2";

/// The dose targets each analytical topology accepts.
///
/// **Exhaustive on purpose.** Adding a `PkModel` variant breaks this `match`, and
/// the person adding it has to state which compartments a dose can be written
/// into. They do *not* have to state whether `W_CMT_DEFAULTED` should fire — the
/// test measures that.
fn case_for(model: PkModel) -> Case {
    let (structural, params, indiv, probe_cmts) = match model {
        // Probe CMT=2 even though there is no second compartment: it must come back
        // *rejected*, which is a measurement. An empty probe list would make this
        // row's `observable == false` true by having asked nothing.
        PkModel::OneCptIv => ("pk one_cpt_iv(cl=CL, v=V)", "", "", vec![2]),
        PkModel::OneCptOral => (
            "pk one_cpt_oral(cl=CL, v=V, ka=KA)",
            KA_P,
            KA_I,
            // CMT=2 is the depot-bypassing central bolus (#350).
            vec![2],
        ),
        PkModel::TwoCptIv => (
            "pk two_cpt_iv(cl=CL, v=V, q=Q, v2=V2)",
            TWO_P,
            TWO_I,
            // CMT=2 is the peripheral.
            vec![2],
        ),
        PkModel::TwoCptOral => (
            "pk two_cpt_oral(cl=CL, v=V, q=Q, v2=V2, ka=KA)",
            "  theta TVKA(1.0, 0.01, 10.0)\n  theta TVQ(3.0, 0.1, 50.0)\n  \
             theta TVV2(80.0, 5.0, 500.0)",
            "  KA = TVKA\n  Q  = TVQ\n  V2 = TVV2",
            vec![2, 3],
        ),
        PkModel::ThreeCptIv => (
            "pk three_cpt_iv(cl=CL, v=V, q=Q, v2=V2, q3=Q3, v3=V3)",
            THREE_P,
            THREE_I,
            vec![2, 3],
        ),
        PkModel::ThreeCptOral => (
            "pk three_cpt_oral(cl=CL, v=V, q=Q, v2=V2, q3=Q3, v3=V3, ka=KA)",
            "  theta TVKA(1.0, 0.01, 10.0)\n  theta TVQ(3.0, 0.1, 50.0)\n  \
             theta TVV2(80.0, 5.0, 500.0)\n  theta TVQ3(1.0, 0.1, 50.0)\n  \
             theta TVV3(120.0, 5.0, 900.0)",
            "  KA = TVKA\n  Q  = TVQ\n  V2 = TVV2\n  Q3 = TVQ3\n  V3 = TVV3",
            vec![2, 3, 4],
        ),
        // Transit and inverse-Gaussian carry two or three *states* but no dose
        // channels at all: the closed form absorbs every dose through the depot and
        // `single_dose_concentration` never reads `dose.cmt`. Probing CMT=2 here is
        // the point — the prediction must come back identical, which is what earns
        // them the suppression.
        PkModel::OneCptTransit => (
            "pk one_cpt_transit(cl=CL, v=V, n=NN, mtt=MTT)",
            TRANSIT_P,
            TRANSIT_I,
            vec![2],
        ),
        PkModel::TwoCptTransit => (
            // `v1`, not `v`, on the two-compartment transit form.
            "pk two_cpt_transit(cl=CL, v1=V, q=Q, v2=V2, n=NN, mtt=MTT)",
            "  theta TVQ(3.0, 0.1, 50.0)\n  theta TVV2(80.0, 5.0, 500.0)\n  \
             theta TVMTT(2.0, 0.1, 20.0)\n  theta TVNN(3.0, 1.0, 20.0)",
            "  Q  = TVQ\n  V2 = TVV2\n  MTT = TVMTT\n  NN = TVNN",
            vec![2, 3],
        ),
        PkModel::OneCptIg => (
            "pk one_cpt_ig(cl=CL, v=V, mat=MAT, cv2=CV2)",
            IG_P,
            IG_I,
            vec![2],
        ),
        PkModel::TwoCptIg => (
            "pk two_cpt_ig(cl=CL, v1=V, q=Q, v2=V2, mat=MAT, cv2=CV2)",
            "  theta TVQ(3.0, 0.1, 50.0)\n  theta TVV2(80.0, 5.0, 500.0)\n  \
             theta TVMAT(2.0, 0.1, 20.0)\n  theta TVCV2(0.5, 0.05, 5.0)",
            "  Q  = TVQ\n  V2 = TVV2\n  MAT = TVMAT\n  CV2 = TVCV2",
            vec![2, 3],
        ),
    };
    Case {
        src: model_src(structural, params, indiv),
        probe_cmts,
    }
}

/// Every analytical topology. Parallel to the `match` above; `case_for`'s
/// exhaustiveness is what guarantees a new variant is noticed, this list is what
/// guarantees it is *run*.
const ALL: [PkModel; 10] = [
    PkModel::OneCptIv,
    PkModel::OneCptOral,
    PkModel::TwoCptIv,
    PkModel::TwoCptOral,
    PkModel::ThreeCptIv,
    PkModel::ThreeCptOral,
    PkModel::OneCptTransit,
    PkModel::TwoCptTransit,
    PkModel::OneCptIg,
    PkModel::TwoCptIg,
];

fn temp(contents: &str, suffix: &str) -> NamedTempFile {
    let mut f = tempfile::Builder::new()
        .suffix(suffix)
        .tempfile()
        .expect("temp file");
    write!(f, "{contents}").expect("write");
    f.flush().expect("flush");
    f
}

fn pop_of(csv: &str) -> Population {
    let f = temp(csv, ".csv");
    read_nonmem_csv(f.path(), None, None).expect("dataset loads")
}

/// A one-subject dataset dosing `dose_cmt`, observing on compartment 1.
fn csv_with_cmt(dose_cmt: usize) -> String {
    let obs: String = OBS_TIMES
        .iter()
        .map(|t| format!("1,{t},5.0,0,.,1,0"))
        .collect::<Vec<_>>()
        .join("\n");
    format!("ID,TIME,DV,EVID,AMT,CMT,MDV\n1,0,.,1,100,{dose_cmt},1\n{obs}\n")
}

/// The same rows with the `CMT` column removed entirely — the #1009 case.
fn csv_without_cmt() -> String {
    let obs: String = OBS_TIMES
        .iter()
        .map(|t| format!("1,{t},5.0,0,.,0"))
        .collect::<Vec<_>>()
        .join("\n");
    format!("ID,TIME,DV,EVID,AMT,MDV\n1,0,.,1,100,1\n{obs}\n")
}

/// Predictions for a dose into `dose_cmt`, or `None` when the model *rejects* that
/// compartment.
///
/// A rejected `CMT` cannot be silently substituted — the user gets an error naming
/// it — so it contributes nothing to observability, and lumping the two cases
/// together would let a rejection masquerade as agreement. Checked through
/// `validate_model_file` first rather than by catching a panic, so the reason is a
/// diagnostic code rather than an unwind.
fn try_preds(src: &str, dose_cmt: usize) -> Option<Vec<f64>> {
    reject_codes(src, dose_cmt).ok()?;
    let csv = csv_with_cmt(dose_cmt);
    let model: CompiledModel = parse_full_model(src).expect("model parses").model;
    Some(
        predict(&model, &pop_of(&csv), &model.default_params)
            .into_iter()
            .map(|p| p.pred)
            .collect(),
    )
}

/// `Ok(())` when the model accepts a dose into `dose_cmt`, `Err(codes)` with the
/// error codes that rejected it otherwise.
///
/// Returning the codes rather than a bare bool is what lets the degenerate test
/// assert *why* a compartment was refused. Checking only `report.valid` let a wrong
/// claim about which code fires sit in a comment unchallenged.
fn reject_codes(src: &str, dose_cmt: usize) -> Result<(), Vec<String>> {
    let m = temp(src, ".ferx");
    let d = temp(&csv_with_cmt(dose_cmt), ".csv");
    let report = validate_model_file(m.path().to_str().unwrap(), Some(d.path().to_str().unwrap()));
    if report.valid {
        return Ok(());
    }
    Err(report
        .diagnostics
        .iter()
        .filter(|x| x.code.starts_with('E'))
        .map(|x| x.code.clone())
        .collect())
}

/// The compartment-1 baseline, which every topology must accept.
fn base_preds(src: &str, model: PkModel) -> Vec<f64> {
    try_preds(src, 1).unwrap_or_else(|| panic!("{model:?}: CMT=1 must be a valid dose target"))
}

/// Largest relative difference between two prediction vectors.
///
/// Folded with an explicit `is_finite` guard on both sides rather than through
/// `f64::max`, which discards `NaN` — a solver returning `NaN` is the likeliest way
/// to break what this measures, and a plain running max would report the *finite*
/// rows' answer and pass (CLAUDE.md, "a green test is not evidence that it can
/// fail").
fn worst_rel_diff(a: &[f64], b: &[f64], case: &str) -> f64 {
    assert_eq!(a.len(), b.len(), "{case}: row count");
    assert!(!a.is_empty(), "{case}: no rows to compare");
    let mut worst = 0.0f64;
    for (j, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!(
            x.is_finite() && y.is_finite(),
            "{case} obs {j}: non-finite prediction ({x}, {y}) — the comparison below \
             would silently ignore it"
        );
        let d = (x - y).abs() / x.abs().max(y.abs()).max(1e-12);
        if d > worst {
            worst = d;
        }
    }
    worst
}

/// Whether `ferx check` reports `W_CMT_DEFAULTED` for this model on a dataset with
/// no `CMT` column. Goes through the public check entry point, so it exercises the
/// same suppression filter `fit()` applies rather than a private predicate.
fn warns_without_cmt(src: &str) -> bool {
    let m = temp(src, ".ferx");
    let d = temp(&csv_without_cmt(), ".csv");
    let report = validate_model_file(m.path().to_str().unwrap(), Some(d.path().to_str().unwrap()));
    report
        .diagnostics
        .iter()
        .any(|x| x.code == "W_CMT_DEFAULTED")
}

#[test]
fn the_warning_fires_exactly_where_the_dose_compartment_is_observable() {
    let mut table: Vec<String> = Vec::new();
    for model in ALL {
        let case = case_for(model);
        // Without this, a row whose probe list is empty reports `observable ==
        // false` by having asked nothing, and agrees with a suppressed predicate for
        // free. Verified by mutation: emptying `OneCptIv`'s list left both tests in
        // this file green.
        assert!(
            !case.probe_cmts.is_empty(),
            "{model:?}: probe list is empty, so `observable` would be false by \
             default rather than by measurement"
        );
        let base = base_preds(&case.src, model);

        // Measured, not declared: is *any* reachable compartment distinguishable
        // from compartment 1 on this topology?
        let mut observable = false;
        let mut worst_overall = 0.0f64;
        let mut rejected: Vec<usize> = Vec::new();
        for &k in &case.probe_cmts {
            let Some(alt) = try_preds(&case.src, k) else {
                // The model refuses this compartment, so no dataset can lose it
                // silently. Recorded, not skipped quietly.
                rejected.push(k);
                continue;
            };
            let worst = worst_rel_diff(&base, &alt, &format!("{model:?} CMT={k}"));
            if worst > worst_overall {
                worst_overall = worst;
            }
            if worst > DISTINGUISHABLE {
                observable = true;
            }
        }

        let warned = warns_without_cmt(&case.src);
        table.push(format!(
            "  {model:?}: probes {:?}, rejected {rejected:?}, worst rel diff \
             {worst_overall:.3e}, observable {observable}, warned {warned}",
            case.probe_cmts
        ));

        assert_eq!(
            warned,
            observable,
            "{model:?}: W_CMT_DEFAULTED must fire exactly when the dose compartment is \
             observable. Measured worst relative difference against CMT=1 over probes \
             {:?} was {worst_overall:.6e} (threshold {DISTINGUISHABLE:.0e}), so \
             `observable` is {observable}, but `ferx check` {}. Either the predicate \
             in `api::validation::cmt_defaulting_is_ambiguous` disagrees with this \
             model's dose routing, or the probe list in `case_for` is wrong.\n\nFull \
             table:\n{}",
            case.probe_cmts,
            if warned { "warned" } else { "did not warn" },
            table.join("\n")
        );
    }

    // The straddle, asserted so this file cannot quietly become a one-sided test:
    // both outcomes must actually occur across the topologies. Without it, a
    // predicate stuck at `true` (or at `false`) passes every row above.
    let warned_any = ALL.iter().any(|&m| warns_without_cmt(&case_for(m).src));
    let suppressed_any = ALL.iter().any(|&m| !warns_without_cmt(&case_for(m).src));
    assert!(
        warned_any && suppressed_any,
        "the topologies must cover both outcomes, else this test cannot fail:\n{}",
        table.join("\n")
    );

    // Printed on every run: the realised separations are the evidence that
    // `DISTINGUISHABLE` separates kinds rather than splitting a continuum. Run with
    // `--nocapture` to read it.
    eprintln!("cmt-defaulting scope, realised:\n{}", table.join("\n"));
}

/// The degenerate half, stated as its own property rather than left implicit in the
/// agreement above: on a topology with no dose channels, a second compartment is
/// either **rejected outright** or **bit-identical** to compartment 1 — never
/// "close". Either outcome means no dataset can lose the compartment silently,
/// which is what earns these models their suppression.
///
/// **Which arm is live, measured:** today all four reject every probe, so the
/// bit-identity comparison is a *guard* rather than an exercised path — worth
/// knowing before trusting it. The rejecting codes are `E_TRANSIT_UNSUPPORTED` and
/// `E_IG_UNSUPPORTED`, asserted below rather than described: an earlier version of
/// this comment claimed `E_DOSE_CMT_OUT_OF_RANGE`, which is what `one_cpt_iv` gives
/// and is not one of these four models at all — `check_dose_compartments` bounds on
/// `n_states`, and transit/IG have 2 or 3 of those, so the range rule cannot be what
/// rejects them. It exists because
/// "rejected" and "accepted but ignored" are the two ways to be safe here and only
/// one of them is currently taken: if a change ever made a transit dose accepted,
/// this reddens on an exact comparison rather than waiting for the difference to
/// clear a threshold. The *exercised* content of this file for those models is the
/// `observable == warned == false` rows in the test above.
#[test]
fn a_topology_with_no_dose_channels_either_rejects_cmt_or_ignores_it_bit_for_bit() {
    let mut outcomes: Vec<String> = Vec::new();
    for model in [
        PkModel::OneCptTransit,
        PkModel::TwoCptTransit,
        PkModel::OneCptIg,
        PkModel::TwoCptIg,
    ] {
        let case = case_for(model);
        let base = base_preds(&case.src, model);
        assert!(
            base.iter().all(|v| v.is_finite()),
            "{model:?}: CMT=1 predictions must be finite before comparing: {base:?}"
        );
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        for &k in &case.probe_cmts {
            match try_preds(&case.src, k) {
                None => {
                    // Assert *why*, not just that. A comment here previously named
                    // `E_DOSE_CMT_OUT_OF_RANGE`, which is what `one_cpt_iv` gives and
                    // is not one of these models: `check_dose_compartments` bounds on
                    // `n_states`, and transit/IG carry 2 or 3 of those, so the range
                    // rule cannot be what refuses them. Only reading `report.valid`
                    // let that wrong claim stand.
                    let codes = reject_codes(&case.src, k).expect_err("just rejected");
                    assert!(
                        codes
                            .iter()
                            .any(|c| c == "E_TRANSIT_UNSUPPORTED" || c == "E_IG_UNSUPPORTED"),
                        "{model:?} CMT={k}: refused, but not by the absorption-model rule \
                         this test is about — got {codes:?}"
                    );
                    outcomes.push(format!("{model:?} CMT={k}: rejected by {codes:?}"));
                }
                Some(alt) => {
                    assert_eq!(
                        bits(&alt),
                        bits(&base),
                        "{model:?}: CMT={k} is accepted, so it must be bit-identical to \
                         CMT=1 — the closed form absorbs every dose through the depot and \
                         never reads `dose.cmt`. Anything in between is a silent \
                         substitution this model is suppressed for not having."
                    );
                    outcomes.push(format!("{model:?} CMT={k}: accepted, bit-identical"));
                }
            }
        }
    }
    assert!(
        !outcomes.is_empty(),
        "no probes ran — the case table lost its transit/IG entries"
    );
    eprintln!("{}", outcomes.join("\n"));
}
