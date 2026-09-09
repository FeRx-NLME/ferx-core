use super::*;

fn parse(src: &str) -> Mfl {
    Mfl::parse(src).unwrap_or_else(|e| panic!("{src:?} should parse: {e}"))
}

fn err(src: &str) -> String {
    match Mfl::parse(src) {
        Ok(m) => panic!("{src:?} should not parse, got {}", m.render()),
        Err(e) => e,
    }
}

/// Parse → render → reparse gives the same feature set, and rendering is a
/// fixed point (the second render equals the first) — the round-trip of the
/// issue's acceptance list.
fn roundtrip(src: &str) -> String {
    let first = parse(src);
    let rendered = first.render();
    let second = Mfl::parse(&rendered)
        .unwrap_or_else(|e| panic!("render of {src:?} → {rendered:?} does not reparse: {e}"));
    assert_eq!(
        first, second,
        "round-trip changed the feature set for {src:?}"
    );
    assert_eq!(
        second.render(),
        rendered,
        "render is not a fixed point for {src:?}"
    );
    rendered
}

#[test]
fn issue_example_round_trips() {
    let src = "ABSORPTION([FO,ZO]); PERIPHERALS(0..1); LAGTIME([OFF,ON])\n\
               COVARIATE?(@IIV, @CONTINUOUS, [pow,lin]); COVARIATE?(@IIV, @CATEGORICAL, cat)\n";
    let rendered = roundtrip(src);
    assert_eq!(
        rendered,
        "ABSORPTION([FO,ZO]);PERIPHERALS(0..1);LAGTIME([OFF,ON]);\
         COVARIATE?(@IIV,@CONTINUOUS,[pow,lin]);COVARIATE?(@IIV,@CATEGORICAL,cat)"
    );
}

#[test]
fn every_feature_category_round_trips() {
    for src in [
        "ABSORPTION(FO)",
        "ABSORPTION(*)",
        "ABSORPTION([INST,FO,ZO,SEQ-ZO-FO,WEIBULL])",
        "ELIMINATION([FO,ZO,MM,MIX-FO-MM])",
        "PERIPHERALS(1)",
        "PERIPHERALS([0,2])",
        "PERIPHERALS(0..2,MET)",
        "PERIPHERALS(1,[DRUG,MET])",
        "TRANSITS(N)",
        "TRANSITS(0..3)",
        "TRANSITS([1,3],NODEPOT)",
        "TRANSITS(2,*)",
        "LAGTIME(ON)",
        "LAGTIME(*)",
        "COVARIATE(CL,WT,pow)",
        "COVARIATE([CL,V],[WT,AGE],[exp,pow])",
        "COVARIATE?(*,*,*)",
        "COVARIATE?(@PK,@CONTINUOUS,*,+)",
        "COVARIATE?(CL,SEX,cat)",
        "ALLOMETRY(WT)",
        "ALLOMETRY(WT,70)",
        "ALLOMETRY(WT,70.5)",
        "DIRECTEFFECT([LINEAR,EMAX])",
        "EFFECTCOMP(*)",
        "INDIRECTEFFECT(EMAX,PRODUCTION)",
        "INDIRECTEFFECT(*,*)",
        "METABOLITE(PSC)",
        "IIV(@PK,EXP)",
        "IIV?([CL,V],[EXP,ADD])",
        "IOV(CL,PROP)",
        "IOV?(*,*)",
        "COVARIANCE(IIV,@PK_IIV)",
        "COVARIANCE?(IOV,[CL,V])",
        "COVARIANCE([IIV,IOV],[CL,V])",
        "COVARIANCE(*,*)",
        "LET(CONTINUOUS,[WT,AGE])",
        "LET(X,WT)",
    ] {
        let rendered = roundtrip(src);
        assert_eq!(rendered, src, "canonical spelling should be stable");
    }
}

#[test]
fn keywords_and_modes_are_case_insensitive_names_are_not() {
    let a = parse("absorption([fo, Zo]); covariate?(Cl, wt, POW)");
    let b = parse("ABSORPTION([FO,ZO]);COVARIATE?(Cl,wt,pow)");
    assert_eq!(a, b);
    // The parameter and covariate names kept their case (ferx deviation).
    match &b.statements[1] {
        Statement::Feature(Feature::Covariate {
            parameters,
            covariates,
            ..
        }) => {
            assert_eq!(parameters, &Operand::Names(vec!["Cl".into()]));
            assert_eq!(covariates, &Operand::Names(vec!["wt".into()]));
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn separators_whitespace_and_comments() {
    let a = parse("  ABSORPTION(FO) ;; \n\n ELIMINATION( FO ) # trailing note\n# whole-line note\nLAGTIME(ON)\n");
    let b = parse("ABSORPTION(FO);ELIMINATION(FO);LAGTIME(ON)");
    assert_eq!(a, b);
    assert!(parse("").statements.is_empty());
    assert!(parse("# only a comment\n;\n").statements.is_empty());
}

#[test]
fn single_element_arrays_normalise_to_bare_values() {
    assert_eq!(parse("ABSORPTION([FO])"), parse("ABSORPTION(FO)"));
    assert_eq!(
        parse("COVARIATE([CL],[WT],[pow])"),
        parse("COVARIATE(CL,WT,pow)")
    );
    // Duplicates in a mode list collapse.
    assert_eq!(parse("LAGTIME([ON,ON,OFF])"), parse("LAGTIME([ON,OFF])"));
}

#[test]
fn counts_expand_inclusive_and_deduplicated() {
    assert_eq!(Counts::Range(0, 2).expand(), vec![0, 1, 2]);
    assert_eq!(Counts::List(vec![2, 0, 2]).expand(), vec![0, 2]);
    assert_eq!(Counts::Single(1).expand(), vec![1]);
}

#[test]
fn wildcard_expands_to_every_mode() {
    assert_eq!(
        Modes::<AbsorptionMode>::Wildcard.expand(),
        AbsorptionMode::ALL.to_vec()
    );
    // `Modes::expand` is the raw enum; the covariate wildcard's *meaning*
    // (the four continuous forms) is applied by coverage and resolve.
    assert_eq!(
        Modes::<CovariateEffect>::Wildcard.expand().len(),
        CovariateEffect::ALL.len()
    );
    assert_eq!(CovariateEffect::CONTINUOUS.len(), 4);
    assert!(!CovariateEffect::CONTINUOUS
        .iter()
        .any(|e| e.is_categorical()));
}

#[test]
fn covariate_operator_defaults_to_multiply() {
    match parse("COVARIATE(CL,WT,pow)").statements.pop() {
        Some(Statement::Feature(Feature::Covariate { op, optional, .. })) => {
            assert_eq!(op, CovariateOp::Multiply);
            assert!(!optional);
        }
        other => panic!("unexpected {other:?}"),
    }
    match parse("COVARIATE?(CL,WT,pow,+)").statements.pop() {
        Some(Statement::Feature(Feature::Covariate { op, optional, .. })) => {
            assert_eq!(op, CovariateOp::Add);
            assert!(optional);
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn features_iterator_skips_lets() {
    let m = parse("LET(X,[A,B]);ABSORPTION(FO);LET(Y,C)");
    assert_eq!(m.features().count(), 1);
    assert_eq!(m.statements.len(), 3);
}

#[test]
fn keyword_of_feature_includes_optional_marker() {
    let m = parse("COVARIATE?(CL,WT,pow);IIV(CL,EXP)");
    let kws: Vec<String> = m.features().map(Feature::keyword).collect();
    assert_eq!(kws, vec!["COVARIATE?", "IIV"]);
}

// --- errors -----------------------------------------------------------------

#[test]
fn unknown_keyword_names_the_known_ones() {
    let e = err("ABSORB(FO)");
    assert!(e.contains("`ABSORB` is not a feature keyword"), "{e}");
    assert!(e.contains("ABSORPTION"), "{e}");
}

#[test]
fn unknown_mode_names_the_set_and_the_known_values() {
    let e = err("ABSORPTION(FAST)");
    assert!(e.contains("`FAST` is not a known absorption mode"), "{e}");
    assert!(e.contains("SEQ-ZO-FO"), "{e}");
    let e = err("COVARIATE(CL,WT,sqrt)");
    assert!(e.contains("`sqrt` is not a known covariate effect"), "{e}");
    assert!(e.contains("piece_lin"), "{e}");
}

#[test]
fn arity_errors_say_what_the_feature_takes() {
    let e = err("ABSORPTION(FO,ZO)");
    assert!(e.contains("ABSORPTION takes one argument"), "{e}");
    assert!(e.contains("found 2 arguments"), "{e}");
    let e = err("COVARIATE(CL,WT)");
    assert!(
        e.contains("COVARIATE takes parameter(s), covariate(s), effect(s)"),
        "{e}"
    );
    let e = err("COVARIANCE(IIV)");
    assert!(
        e.contains("COVARIANCE takes IIV or IOV, then parameter(s)"),
        "{e}"
    );
}

#[test]
fn optional_marker_is_only_for_the_four_exploratory_features() {
    let e = err("ABSORPTION?(FO)");
    assert!(e.contains("`ABSORPTION?` is not valid"), "{e}");
    assert!(e.contains("COVARIATE, IIV, IOV and COVARIANCE"), "{e}");
}

#[test]
fn range_and_count_errors() {
    let e = err("PERIPHERALS(2..0)");
    assert!(e.contains("runs backwards"), "{e}");
    let e = err("PERIPHERALS(FO)");
    assert!(
        e.contains("expected a count, a range `a..b`, or an array of counts"),
        "{e}"
    );
    let e = err("PERIPHERALS([1,x])");
    assert!(e.contains("`x` is not a count"), "{e}");
    let e = err("PERIPHERALS(1.5)");
    assert!(
        e.contains("`1.5` is not a non-negative integer count"),
        "{e}"
    );
}

#[test]
fn counts_are_bounded_at_parse_time() {
    // A range is materialised by `Counts::expand`, so an unbounded one is an
    // allocation of the user's choosing at config-load time. The bound is on
    // every count, so it covers a range's ends, a single and an array alike.
    let e = err("PERIPHERALS(0..4294967295)");
    assert!(
        e.contains("`4294967295` is above the largest compartment count accepted (64)"),
        "{e}"
    );
    let e = err("TRANSITS(65)");
    assert!(e.contains("`65` is above"), "{e}");
    let e = err("PERIPHERALS([1,65])");
    assert!(e.contains("`65` is above"), "{e}");
    parse("PERIPHERALS(0..64)");
    assert_eq!(MAX_COUNT, 64);
}

#[test]
fn syntax_errors_are_located() {
    let e = err("ABSORPTION(FO");
    assert!(e.contains("expected `)`"), "{e}");
    let e = err("ABSORPTION FO");
    assert!(e.contains("expected `(`"), "{e}");
    let e = err("ABSORPTION(FO) ELIMINATION(FO)");
    assert!(
        e.contains("expected `;` or a newline after a statement"),
        "{e}"
    );
    let e = err("COVARIATE(CL,[WT,pow)");
    assert!(e.contains("expected `,` or `]`"), "{e}");
    let e = err("COVARIATE(@,WT,pow)");
    assert!(e.contains("expected a symbol name after `@`"), "{e}");
    let e = err("ABSORPTION(FO) $");
    assert!(e.contains("unexpected character `$`"), "{e}");
    let e = err("LET(X)");
    assert!(e.contains("in LET: expected `,`"), "{e}");
    let e = err("LET(X,[])");
    assert!(e.contains("an empty array `[]` names nothing"), "{e}");
}

#[test]
fn mandatory_covariate_effects_must_be_explicit() {
    // Pharmpy: "Mandatory effects need to be explicit (not '*')".
    let e = err("COVARIATE(CL,WT,*)");
    assert!(e.contains("needs explicit effects, not `*`"), "{e}");
    parse("COVARIATE?(CL,WT,*)");
}

#[test]
fn transits_n_takes_no_depot_option() {
    // Pharmpy's grammar: `transits: ... (n | (_counts ["," _depot_option]))`.
    let e = err("TRANSITS(N,DEPOT)");
    assert!(e.contains("`TRANSITS(N)` takes no second argument"), "{e}");
    parse("TRANSITS(N)");
    parse("TRANSITS(1,DEPOT)");
}

#[test]
fn covariance_level_is_a_mode_argument() {
    // The grammar's first argument is `values | wildcard`: a level, an array
    // of levels, or `*`.
    let e = err("COVARIANCE(RUV,[CL,V])");
    assert!(e.contains("`RUV` is not a known variability level"), "{e}");
    assert!(e.contains("known: IIV, IOV"), "{e}");
    assert_eq!(
        parse("COVARIANCE([IIV],[CL,V])"),
        parse("COVARIANCE(IIV,[CL,V])")
    );
    match parse("COVARIANCE(*,[CL,V])").statements.pop() {
        Some(Statement::Feature(Feature::Covariance { level, .. })) => {
            assert_eq!(level.expand(), VariabilityLevel::ALL.to_vec());
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn covariate_fourth_argument_must_be_an_operator() {
    let e = err("COVARIATE(CL,WT,pow,lin)");
    assert!(e.contains("the fourth argument is the operator"), "{e}");
}

#[test]
fn allometry_reference_must_be_numeric() {
    let e = err("ALLOMETRY(WT,seventy)");
    assert!(e.contains("reference value must be a number"), "{e}");
}
