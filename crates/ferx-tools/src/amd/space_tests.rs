use super::*;
use crate::search::Mfl;

fn mfl(src: &str) -> Mfl {
    Mfl::parse(src).expect("fixture MFL")
}

/// The full AMD space, one statement per step plus a two-level covariance.
const FULL: &str = "ABSORPTION([INST,FO]);PERIPHERALS(0..1);LAGTIME([OFF,ON]);\
                    IIV?([CL,V],EXP);COVARIANCE?(IIV,[CL,V]);\
                    IOV?([CL],EXP);\
                    ALLOMETRY(WT,70);\
                    COVARIATE?(@IIV,@CONTINUOUS,[pow,lin])";

/// Every statement lands in exactly one step's subspace, and every subspace
/// holds only what its tool accepts.
///
/// The assertion is on the *partition*, not on a spot check: the six
/// subspaces' feature counts must sum to the space's own, which is what fails
/// if a future feature kind is routed to two steps or to none.
#[test]
fn every_statement_lands_in_exactly_one_step() {
    let space = mfl(FULL);
    let total = space.features().count();
    let mut sum = 0;
    for step in Step::ALL {
        let sub = subspace(&space, step).unwrap();
        sum += sub.features().count();
        for feature in sub.features() {
            // `COVARIANCE` has no single owner — it is narrowed per level —
            // so it belongs to the variability steps and to nothing else.
            let expected = owner(feature).unwrap().unwrap_or(step);
            assert_eq!(
                expected,
                step,
                "{} was routed to {}",
                feature.keyword(),
                step.label()
            );
            if owner(feature).unwrap().is_none() {
                assert!(
                    matches!(step, Step::Iivsearch | Step::Iovsearch),
                    "a covariance reached {}",
                    step.label()
                );
            }
        }
    }
    assert_eq!(sum, total, "the subspaces do not partition the space");
}

/// The structural subspace is exactly the structural statements — and
/// `modelsearch` accepts it, which is the property that matters: a subspace
/// this module builds must be readable by the tool it is built for.
#[test]
fn the_structural_subspace_is_what_modelsearch_takes() {
    let sub = subspace(&mfl(FULL), Step::Structural).unwrap();
    assert_eq!(
        sub.render(),
        "ABSORPTION([INST,FO]);PERIPHERALS(0..1);LAGTIME([OFF,ON])"
    );
    crate::modelsearch::structure::space_features(&sub)
        .expect("modelsearch reads its own subspace");
}

/// The covariate subspace is what covsearch takes, and nothing structural
/// travels with it.
#[test]
fn the_covariate_subspace_carries_no_structural_statement() {
    let sub = subspace(&mfl(FULL), Step::Covariates).unwrap();
    assert_eq!(sub.features().count(), 1);
    assert_eq!(sub.render(), "COVARIATE?(@IIV,@CONTINUOUS,[pow,lin])");
}

/// `COVARIANCE(*, …)` names both levels, so it is narrowed rather than given
/// to one tool or dropped: `iivsearch` sees the η half, `iovsearch` the κ half,
/// and neither is handed a statement mentioning the other's level (which both
/// tools refuse by name).
#[test]
fn a_two_level_covariance_is_narrowed_to_each_level() {
    let space = mfl("IIV?([CL,V],EXP);IOV?([CL],EXP);COVARIANCE?(*,[CL,V])");
    let iiv = subspace(&space, Step::Iivsearch).unwrap();
    let iov = subspace(&space, Step::Iovsearch).unwrap();
    assert_eq!(iiv.render(), "IIV?([CL,V],EXP);COVARIANCE?(IIV,[CL,V])");
    assert_eq!(iov.render(), "IOV?(CL,EXP);COVARIANCE?(IOV,[CL,V])");
    // And a single-level covariance goes to its own level only.
    let one = mfl("IIV?([CL,V],EXP);COVARIANCE?(IIV,[CL,V])");
    assert_eq!(
        subspace(&one, Step::Iovsearch).unwrap().features().count(),
        0
    );
}

/// Every subspace re-parses to itself, so the rendered text a tool is handed
/// describes the same features the file did.
#[test]
fn every_subspace_round_trips_through_its_rendering() {
    let space = mfl(FULL);
    for step in Step::ALL {
        let sub = subspace(&space, step).unwrap();
        if sub.statements.is_empty() {
            continue;
        }
        assert_eq!(
            Mfl::parse(&sub.render()).unwrap(),
            sub,
            "{} subspace does not round-trip",
            step.label()
        );
    }
}

/// A `LET` is a definition the statements refer to, so it travels into every
/// subspace that has a statement — and into none that does not, since a
/// subspace of definitions alone is not a search and would make an empty step
/// look runnable.
#[test]
fn let_travels_with_the_statements_but_never_alone() {
    let space = mfl("LET(P,[CL,V]);IIV?(@P,EXP)");
    let iiv = subspace(&space, Step::Iivsearch).unwrap();
    assert_eq!(iiv.render(), "LET(P,[CL,V]);IIV?(@P,EXP)");
    let structural = subspace(&space, Step::Structural).unwrap();
    assert!(
        structural.statements.is_empty(),
        "a LET-only subspace must be empty, got `{}`",
        structural.render()
    );
}

/// A statement no AMD step runs is refused by name, before anything is
/// fitted — never routed to the nearest step and never dropped.
#[test]
fn a_pd_statement_is_refused_by_name() {
    let space = mfl("DIRECTEFFECT(LINEAR)");
    let error = check(&space).unwrap_err();
    assert!(
        error.contains("DIRECTEFFECT") && error.contains("structsearch"),
        "{error}"
    );
    // And through the subspace path too, so no caller can bypass the check.
    assert!(subspace(&space, Step::Structural).is_err());
}
