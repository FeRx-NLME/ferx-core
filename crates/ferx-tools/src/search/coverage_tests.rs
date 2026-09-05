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
        "ELIMINATION(FO)",
        "PERIPHERALS(0..2)",
        "PERIPHERALS(1,DRUG)",
        "TRANSITS(N)",
        "TRANSITS(0..3)",
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
    assert_eq!(gaps("ELIMINATION(MM)"), vec!["ELIMINATION(MM)"]);
    assert_eq!(gaps("ELIMINATION(ZO)"), vec!["ELIMINATION(ZO)"]);
    assert_eq!(
        gaps("ELIMINATION(MIX-FO-MM)"),
        vec!["ELIMINATION(MIX-FO-MM)"]
    );
    assert_eq!(gaps("ABSORPTION(SEQ-ZO-FO)"), vec!["ABSORPTION(SEQ-ZO-FO)"]);
    // ZO and WEIBULL are ODE-only input functions, not `pk` templates: the
    // same class as SEQ-ZO-FO, and a search that admits them would fail per
    // candidate instead of at load.
    assert_eq!(gaps("ABSORPTION(ZO)"), vec!["ABSORPTION(ZO)"]);
    assert_eq!(gaps("ABSORPTION(WEIBULL)"), vec!["ABSORPTION(WEIBULL)"]);
    assert_eq!(gaps("ABSORPTION([FO,ZO])"), vec!["ABSORPTION(ZO)"]);
    assert_eq!(gaps("DIRECTEFFECT(EMAX)"), vec!["DIRECTEFFECT(...)"]);
    assert_eq!(gaps("EFFECTCOMP(*)"), vec!["EFFECTCOMP(...)"]);
    assert_eq!(
        gaps("INDIRECTEFFECT(LINEAR,DEGRADATION)"),
        vec!["INDIRECTEFFECT(...)"]
    );
}

#[test]
fn wildcards_are_checked_against_their_full_expansion() {
    assert_eq!(
        gaps("ELIMINATION(*)"),
        vec![
            "ELIMINATION(ZO)",
            "ELIMINATION(MM)",
            "ELIMINATION(MIX-FO-MM)"
        ]
    );
    assert_eq!(
        gaps("ABSORPTION(*)"),
        vec![
            "ABSORPTION(ZO)",
            "ABSORPTION(SEQ-ZO-FO)",
            "ABSORPTION(WEIBULL)"
        ]
    );
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
    assert_eq!(gaps("METABOLITE(PSC)"), vec!["METABOLITE(...)"]);
    assert_eq!(gaps("COVARIATE(CL,SEX,cat2)"), vec!["COVARIATE(..., cat2)"]);
    assert_eq!(
        gaps("COVARIATE(CL,WT,custom)"),
        vec!["COVARIATE(..., custom)"]
    );
    assert_eq!(gaps("COVARIATE(CL,WT,pow,+)"), vec!["COVARIATE(..., +)"]);
    assert_eq!(gaps("IOV(CL,EXP)"), vec!["IOV(...)"]);
    assert_eq!(gaps("COVARIANCE(IOV,[CL,V])"), vec!["COVARIANCE(IOV, ...)"]);
    assert_eq!(gaps("COVARIANCE(IIV,[CL,V])"), Vec::<String>::new());
    // An array or `*` in the level slot reaches the coverage table, not the
    // parser: the IOV member is the named gap, the IIV member is fine.
    assert_eq!(
        gaps("COVARIANCE([IIV,IOV],[CL,V])"),
        vec!["COVARIANCE(IOV, ...)"]
    );
    assert_eq!(gaps("COVARIANCE(*,[CL,V])"), vec!["COVARIANCE(IOV, ...)"]);
}

#[test]
fn every_gap_is_reported_once_with_a_reason_and_the_docs_link() {
    let mfl =
        Mfl::parse("ELIMINATION([MM,MM]);ELIMINATION(MM);IOV(CL,EXP);IOV(V,ADD)").expect("parses");
    let e = check_coverage(&mfl).expect_err("has gaps");
    assert_eq!(e.gaps.len(), 2, "{e}");
    let text = e.to_string();
    assert!(
        text.starts_with("the search space asks for 2 features ferx cannot build"),
        "{text}"
    );
    assert!(
        text.contains("ELIMINATION(MM): only first-order elimination"),
        "{text}"
    );
    assert!(
        text.contains("IOV(...): inter-occasion variability"),
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
    let mfl = Mfl::parse("ELIMINATION(MM)").expect("parses");
    let text = check_coverage(&mfl).expect_err("gap").to_string();
    assert!(
        text.starts_with("the search space asks for 1 feature ferx"),
        "{text}"
    );
}
