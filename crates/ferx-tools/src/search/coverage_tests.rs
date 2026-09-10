use super::*;

fn gaps(src: &str) -> Vec<String> {
    let mfl = Mfl::parse(src).expect("parses");
    match check_coverage(&mfl) {
        Ok(()) => vec![],
        Err(e) => e.gaps.into_iter().map(|g| g.feature).collect(),
    }
}

#[test]
fn the_epic_table_supported_rows_pass() {
    for src in [
        "ABSORPTION([INST,FO])",
        // #1257: the ODE candidate family.
        "ABSORPTION([ZO,WEIBULL])",
        "ELIMINATION(*)",
        "PERIPHERALS(0..2)",
        "PERIPHERALS(1,DRUG)",
        "TRANSITS(N)",
        "TRANSITS(0..3,NODEPOT)",
        "TRANSITS(1,NODEPOT)",
        "LAGTIME(*)",
        "COVARIATE?(@IIV,@CONTINUOUS,*)",
        "COVARIATE(CL,SEX,cat)",
        "COVARIATE([CL,V],[WT,AGE],[lin,piece_lin,exp,pow])",
        "ALLOMETRY(WT,70)",
        "IIV(@PK,[EXP,ADD,PROP])",
        "COVARIANCE(IIV,@PK_IIV)",
    ] {
        assert!(
            gaps(src).is_empty(),
            "{src} should be supported: {:?}",
            gaps(src)
        );
    }
}

#[test]
fn the_issue_gaps_are_hard_errors_naming_the_feature() {
    // Every elimination and every absorption but one is buildable since
    // #1257; the exception is the one that is a *disposition* rather than an
    // input term.
    assert_eq!(gaps("ELIMINATION(MM)"), Vec::<String>::new());
    assert_eq!(gaps("ELIMINATION(ZO)"), Vec::<String>::new());
    assert_eq!(gaps("ELIMINATION(MIX-FO-MM)"), Vec::<String>::new());
    assert_eq!(gaps("ABSORPTION(ZO)"), Vec::<String>::new());
    assert_eq!(gaps("ABSORPTION(WEIBULL)"), Vec::<String>::new());
    assert_eq!(gaps("ABSORPTION(SEQ-ZO-FO)"), vec!["ABSORPTION(SEQ-ZO-FO)"]);
    assert_eq!(
        gaps("ABSORPTION([FO,SEQ-ZO-FO])"),
        vec!["ABSORPTION(SEQ-ZO-FO)"]
    );
    assert_eq!(gaps("DIRECTEFFECT(EMAX)"), vec!["DIRECTEFFECT(...)"]);
    assert_eq!(gaps("EFFECTCOMP(*)"), vec!["EFFECTCOMP(...)"]);
    assert_eq!(
        gaps("INDIRECTEFFECT(LINEAR,DEGRADATION)"),
        vec!["INDIRECTEFFECT(...)"]
    );
}

#[test]
fn wildcards_are_checked_against_their_full_expansion() {
    assert_eq!(gaps("ABSORPTION(*)"), vec!["ABSORPTION(SEQ-ZO-FO)"]);
    assert_eq!(gaps("TRANSITS(1,*)"), vec!["TRANSITS(n, DEPOT)"]);
    assert_eq!(gaps("PERIPHERALS(1,*)"), vec!["PERIPHERALS(n, MET)"]);
    assert_eq!(gaps("IIV(CL,*)"), vec!["IIV(..., LOG)", "IIV(..., RE_LOG)"]);
    assert_eq!(
        gaps("COVARIATE?(CL,WT,*)"),
        Vec::<String>::new(),
        "the covariate wildcard is the four continuous forms, all supported"
    );
}

#[test]
fn the_remaining_gap_rows() {
    assert_eq!(gaps("PERIPHERALS(3)"), vec!["PERIPHERALS(3)"]);
    assert_eq!(
        gaps("PERIPHERALS(0..4)"),
        vec!["PERIPHERALS(3)", "PERIPHERALS(4)"]
    );
    assert_eq!(gaps("TRANSITS(2,DEPOT)"), vec!["TRANSITS(n, DEPOT)"]);
    // Omitting the option is Pharmpy's DEPOT, not a way past the gap; `N`
    // has no option and is NODEPOT by Pharmpy's own grammar.
    assert_eq!(gaps("TRANSITS(0..3)"), vec!["TRANSITS(n, DEPOT)"]);
    assert_eq!(gaps("TRANSITS(2)"), vec!["TRANSITS(n, DEPOT)"]);
    assert_eq!(gaps("TRANSITS(N)"), Vec::<String>::new());
    assert_eq!(gaps("METABOLITE(PSC)"), vec!["METABOLITE(...)"]);
    // `cat2` became `categorical2` in #1312, so it is no longer a gap.
    assert_eq!(gaps("COVARIATE(CL,SEX,cat2)"), Vec::<String>::new());
    assert_eq!(
        gaps("COVARIATE(CL,WT,custom)"),
        vec!["COVARIATE(..., custom)"]
    );
    assert_eq!(gaps("COVARIATE(CL,WT,pow,+)"), vec!["COVARIATE(..., +)"]);
    // IOV is searchable since #1183 — in the exponential form only, which is
    // the form `[iov]` estimates; both covariance levels are blocks.
    assert_eq!(gaps("IOV(CL,EXP)"), Vec::<String>::new());
    assert_eq!(gaps("IOV(CL,ADD)"), vec!["IOV(..., ADD)"]);
    assert_eq!(gaps("COVARIANCE(IOV,[CL,V])"), Vec::<String>::new());
    assert_eq!(gaps("COVARIANCE(IIV,[CL,V])"), Vec::<String>::new());
    assert_eq!(gaps("COVARIANCE([IIV,IOV],[CL,V])"), Vec::<String>::new());
    assert_eq!(gaps("COVARIANCE(*,[CL,V])"), Vec::<String>::new());
}

#[test]
fn every_gap_is_reported_once_with_a_reason_and_the_docs_link() {
    let mfl = Mfl::parse(
        "ABSORPTION([SEQ-ZO-FO,SEQ-ZO-FO]);ABSORPTION(SEQ-ZO-FO);IOV(CL,ADD);IOV(V,ADD)",
    )
    .expect("parses");
    let e = check_coverage(&mfl).expect_err("has gaps");
    assert_eq!(e.gaps.len(), 2, "{e}");
    let text = e.to_string();
    assert!(
        text.starts_with("the search space asks for 2 features ferx cannot build"),
        "{text}"
    );
    assert!(
        text.contains("ABSORPTION(SEQ-ZO-FO): sequential zero-order"),
        "{text}"
    );
    assert!(
        text.contains("IOV(..., ADD): `ferx-core::edit` writes a κ inside the exponential"),
        "{text}"
    );
    assert!(text.ends_with(COVERAGE_DOCS), "{text}");
    assert!(
        text.contains("#coverage"),
        "the link must land on the table: {text}"
    );
    // The `String` conversion is what `?` inside the loader uses.
    let s: String = e.into();
    assert_eq!(s, text);
}

#[test]
fn singular_grammar_for_one_gap() {
    let mfl = Mfl::parse("ABSORPTION(SEQ-ZO-FO)").expect("parses");
    let text = check_coverage(&mfl).expect_err("gap").to_string();
    assert!(
        text.starts_with("the search space asks for 1 feature ferx"),
        "{text}"
    );
}
