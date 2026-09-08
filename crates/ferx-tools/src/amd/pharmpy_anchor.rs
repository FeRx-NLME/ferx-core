//! The Pharmpy anchor for the pipeline's sequencing and its space split
//! (#1184).
//!
//! AMD adds no numerics: every estimate it reports comes from a tool that has
//! its own external anchor (#1180–#1183). What it *does* own is the
//! orchestration — which tools run, in what order, and which statements of the
//! one search space each of them is handed — and that is what this anchors,
//! against Pharmpy 2.2.0's own `amd`:
//!
//! * [`get_subtool_order`][amd] for every strategy, against [`Strategy::order`];
//! * for a corpus of AMD-shaped spaces, `ss.filter("pk")` against
//!   [`space::subspace`] for the structural step, and Pharmpy's covariate
//!   statements against the covariate step's subspace.
//!
//! `tools/pharmpy-amd-anchor/run.sh` regenerates
//! `tests/data/amd_pharmpy_anchor.json`; nothing is fitted, so it takes a
//! second and needs no NONMEM.
//!
//! Two deliberate divergences are asserted rather than papered over, so one
//! that quietly disappears is a red test just as a new one would be — see
//! [`the_default_fill_divergence_is_real`] and
//! [`a_let_is_resolved_later_than_pharmpy_resolves_it`].
//!
//! [amd]: https://github.com/pharmpy/pharmpy/blob/main/src/pharmpy/tools/amd/run.py

use std::collections::BTreeSet;

use super::space::subspace;
use super::{Step, Strategy};
use crate::search::Mfl;
use serde::Deserialize;

const ANCHOR: &str = include_str!("../../tests/data/amd_pharmpy_anchor.json");

#[derive(Deserialize)]
struct Anchor {
    pharmpy_version: String,
    orders: std::collections::BTreeMap<String, Vec<String>>,
    spaces: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    mfl: String,
    /// `ss.filter("pk")` — the space's own structural statements, with every
    /// unstated PK feature filled in at its base value.
    pk: String,
    /// The space's own covariate statements.
    covariate: String,
    allometry: Option<Allometry>,
    /// What `amd` hands `modelsearch`: [`pk`](Case::pk), or Pharmpy's whole
    /// default structural space when the file states none.
    modelsearch: String,
    /// Likewise for `covsearch`.
    covsearch: String,
}

#[derive(Deserialize)]
struct Allometry {
    covariate: String,
    reference: f64,
}

/// The statements Pharmpy writes for a PK feature the file did **not**
/// mention: the base model's own value, contributing no alternative.
///
/// Pharmpy's `ModelFeatures` is a total record of the model's structure, so a
/// filtered space always names all five PK features; ferx's subspace is the
/// file's statements and nothing else. The two say the same thing — "this
/// feature is not searched" — so the fillers are dropped before the sets are
/// compared, and *only* when ferx's subspace is silent about that keyword,
/// which is what keeps a genuinely dropped statement red.
const FILLERS: [&str; 5] = [
    "ELIMINATION(FO)",
    "TRANSITS(0)",
    "LAGTIME(OFF)",
    "PERIPHERALS(0)",
    "ABSORPTION(INST)",
];

fn anchor() -> Anchor {
    serde_json::from_str(ANCHOR).expect("the Pharmpy AMD anchor parses")
}

/// One MFL rendering as a set of statements, upper-cased so `pow` and `POW`
/// compare equal (the two sides render effect names in different case).
fn statements(mfl: &str) -> BTreeSet<String> {
    mfl.split(';')
        .map(|s| s.trim().to_ascii_uppercase().replace(' ', ""))
        .filter(|s| !s.is_empty())
        .collect()
}

fn keyword(statement: &str) -> &str {
    statement
        .split_once('(')
        .map(|(k, _)| k.trim_end_matches('?'))
        .unwrap_or(statement)
}

/// `pharmpy` minus the fillers for keywords `ferx` does not mention.
fn without_fillers(pharmpy: &BTreeSet<String>, ferx: &BTreeSet<String>) -> BTreeSet<String> {
    let mentioned: BTreeSet<&str> = ferx.iter().map(|s| keyword(s)).collect();
    pharmpy
        .iter()
        .filter(|s| !(FILLERS.contains(&s.as_str()) && !mentioned.contains(keyword(s))))
        .cloned()
        .collect()
}

fn ferx_subspace(mfl: &str, step: Step) -> BTreeSet<String> {
    let parsed = Mfl::parse(mfl).unwrap_or_else(|e| panic!("ferx cannot parse `{mfl}`: {e}"));
    statements(&subspace(&parsed, step).expect("subspace").render())
}

/// The anchor is the version it was generated against — a regenerated file
/// from a different Pharmpy is a different experiment, and should be read as
/// one.
#[test]
fn the_anchor_names_the_pharmpy_it_came_from() {
    assert_eq!(anchor().pharmpy_version, "2.2.0");
}

/// Every strategy runs the same components in the same order Pharmpy runs
/// them in. This is the pipeline's whole contract.
#[test]
fn every_strategy_order_matches_pharmpy() {
    let anchor = anchor();
    let ferx = [
        ("default", Strategy::Default),
        ("reevaluation", Strategy::Reevaluation),
        ("SIR", Strategy::Sir),
        ("SRI", Strategy::Sri),
        ("RSI", Strategy::Rsi),
    ];
    assert_eq!(
        anchor.orders.len(),
        ferx.len(),
        "Pharmpy offers strategies ferx does not: {:?}",
        anchor.orders.keys().collect::<Vec<_>>()
    );
    for (name, strategy) in ferx {
        let theirs = anchor
            .orders
            .get(name)
            .unwrap_or_else(|| panic!("Pharmpy has no `{name}` strategy"));
        let ours: Vec<String> = strategy
            .order()
            .iter()
            .map(|s| s.label().to_string())
            .collect();
        assert_eq!(&ours, theirs, "the {name} strategy runs a different order");
        assert_eq!(
            strategy.label(),
            name,
            "the strategy is spelled differently"
        );
    }
}

/// The structural step is handed exactly the statements Pharmpy hands its
/// own, for every space in the corpus that states any.
///
/// A space that states none is Pharmpy's default-fill case, asserted
/// separately as a divergence.
#[test]
fn the_structural_subspace_matches_pharmpys() {
    for case in anchor().spaces {
        if case.pk.is_empty() {
            continue;
        }
        let ours = ferx_subspace(&case.mfl, Step::Structural);
        let theirs = without_fillers(&statements(&case.pk), &ours);
        assert_eq!(ours, theirs, "structural subspace of `{}`", case.mfl);
        // And with the defaults filled in, `amd` hands its modelsearch the
        // same thing — the case where the two paths agree.
        assert_eq!(
            without_fillers(&statements(&case.modelsearch), &ours),
            theirs,
            "`{}`: amd narrowed the space it filtered",
            case.mfl
        );
    }
}

/// The covariate step is handed exactly the covariate statements, and the
/// allometry step exactly the ALLOMETRY one — neither travels with the other,
/// and neither reaches the structural step.
#[test]
fn the_covariate_and_allometry_subspaces_match_pharmpys() {
    for case in anchor().spaces {
        let ours = ferx_subspace(&case.mfl, Step::Covariates);
        let theirs = statements(&case.covariate);
        if case.mfl.contains("LET(") {
            // Pharmpy resolves a LET at parse time; ferx resolves it against
            // the model. Asserted as a divergence below.
            continue;
        }
        assert_eq!(ours, theirs, "covariate subspace of `{}`", case.mfl);

        let allometry = ferx_subspace(&case.mfl, Step::Allometry);
        match &case.allometry {
            Some(a) => {
                let expected = format!("ALLOMETRY({},{})", a.covariate, a.reference as i64);
                assert_eq!(
                    allometry,
                    statements(&expected),
                    "allometry subspace of `{}`",
                    case.mfl
                );
            }
            None => assert!(
                allometry.is_empty(),
                "`{}`: ferx found an ALLOMETRY statement Pharmpy did not",
                case.mfl
            ),
        }
        // The structural step never sees a covariate or an allometry
        // statement, which is what every tool of the epic refuses by name.
        let structural = ferx_subspace(&case.mfl, Step::Structural);
        assert!(
            structural
                .iter()
                .all(|s| !s.starts_with("COVARIATE") && !s.starts_with("ALLOMETRY")),
            "`{}`: a covariate statement reached the structural step",
            case.mfl
        );
    }
}

/// **Divergence.** Where the file's space says nothing about a step, Pharmpy
/// substitutes a whole default search space; ferx skips the step and says so.
///
/// The two behaviours are not reconcilable by configuration, so the anchor
/// asserts the difference is real: a corpus case with no PK statement gets a
/// five-statement structural space from Pharmpy and an empty one from ferx.
/// ferx's choice is the epic's: a search the user did not ask for is not a
/// default, and a step whose space is silent is reported as skipped with its
/// reason rather than run on a space invented for it.
#[test]
fn the_default_fill_divergence_is_real() {
    let mut seen = 0;
    for case in anchor().spaces {
        if !case.pk.is_empty() {
            continue;
        }
        seen += 1;
        assert!(
            ferx_subspace(&case.mfl, Step::Structural).is_empty(),
            "`{}`: ferx built a structural subspace from nothing",
            case.mfl
        );
        assert!(
            statements(&case.modelsearch).len() >= 5,
            "`{}`: Pharmpy no longer fills a default structural space — the divergence is gone \
             and this test should be replaced by an equality",
            case.mfl
        );
    }
    assert!(
        seen > 0,
        "the corpus no longer covers the default-fill case"
    );

    // The same on the covariate side: no COVARIATE statement, and Pharmpy
    // still hands covsearch its two exploratory defaults.
    let mut seen = 0;
    for case in anchor().spaces {
        if !case.covariate.is_empty() {
            continue;
        }
        seen += 1;
        assert!(ferx_subspace(&case.mfl, Step::Covariates).is_empty());
        assert!(statements(&case.covsearch).len() >= 2, "`{}`", case.mfl);
    }
    assert!(
        seen > 0,
        "the corpus no longer covers a space with no COVARIATE"
    );
}

/// **Divergence.** Pharmpy resolves a `LET` when it parses the space; ferx
/// carries the definition into the subspace and resolves it against the base
/// model, which is what lets `@IIV` mean "the η this step's parent actually
/// has" rather than the η the file's author had in mind.
///
/// The two agree on the *resolved* statement, so the assertion is that the
/// definition travels with the statement that refers to it and that Pharmpy
/// has already substituted it.
#[test]
fn a_let_is_resolved_later_than_pharmpy_resolves_it() {
    let case = anchor()
        .spaces
        .into_iter()
        .find(|c| c.mfl.contains("LET("))
        .expect("the corpus covers a LET");
    let ours = ferx_subspace(&case.mfl, Step::Covariates);
    let theirs = statements(&case.covariate);
    assert_ne!(ours, theirs, "the LET divergence is gone");
    assert!(
        ours.iter().any(|s| s.starts_with("LET(")),
        "the definition did not travel with its statement: {ours:?}"
    );
    assert!(
        !theirs.iter().any(|s| s.starts_with("LET(")),
        "Pharmpy no longer substitutes the definition: {theirs:?}"
    );
}
