use super::*;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{Population, Subject};

/// A two-compartment oral base with IIV on three of five PK parameters, one
/// non-PK individual parameter, and three declared covariates of both kinds.
const MODEL: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV1(40.0, 1.0, 500.0)
  theta TVQ(8.0, 0.1, 100.0)
  theta TVV2(80.0, 1.0, 500.0)
  theta TVKA(1.0, 0.01, 10.0)
  theta TVF(0.8, 0.1, 1.0)
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  omega ETA_KA ~ 0.20
  sigma PROP_ERR ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
  KA = TVKA * exp(ETA_KA)
  F  = TVF
  # a derived quantity nobody searches over
  KE = CL / V1

[structural_model]
  pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA, f=F)

[covariates]
  WT   continuous
  SEX  categorical(levels = [0, 1])
  CRCL continuous

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

fn population(covariates: &[&str]) -> Population {
    Population {
        subjects: vec![Subject {
            id: "1".into(),
            ..Default::default()
        }],
        covariate_names: covariates.iter().map(|c| c.to_string()).collect(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

fn context_for(model: &str, covariates: &[&str]) -> ModelContext {
    let parsed = parse_full_model(model).expect("fixture parses");
    let text = ModelText::parse(model).expect("fixture is editable");
    ModelContext::from_model(&parsed, &text, &population(covariates)).expect("context")
}

fn ctx() -> ModelContext {
    context_for(MODEL, &["WT", "SEX", "CRCL"])
}

fn resolved(src: &str, ctx: &ModelContext) -> Resolved {
    let mfl = Mfl::parse(src).expect("parses");
    resolve(&mfl, ctx).unwrap_or_else(|e| panic!("{src} should resolve: {e}"))
}

fn resolve_err(src: &str, ctx: &ModelContext) -> String {
    let mfl = Mfl::parse(src).expect("parses");
    match resolve(&mfl, ctx) {
        Ok(r) => panic!("{src} should not resolve, got {}", r.mfl.render()),
        Err(e) => e,
    }
}

fn names(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

// --- the context --------------------------------------------------------------

#[test]
fn context_reads_parameters_iiv_template_and_covariates() {
    let c = ctx();
    assert_eq!(
        c.parameters,
        names(&["CL", "V1", "Q", "V2", "KA", "F", "KE"])
    );
    assert_eq!(c.iiv, names(&["CL", "V1", "KA"]));
    let t = c.template.as_ref().expect("pk line");
    assert_eq!(t.name, "two_cpt_oral");
    assert_eq!(t.parameters(), names(&["CL", "V1", "Q", "V2", "KA", "F"]));
    assert_eq!(
        c.covariates.as_ref().expect("[covariates] block"),
        &vec![
            ("WT".to_string(), CovariateKind::Continuous),
            ("SEX".to_string(), CovariateKind::Categorical),
            ("CRCL".to_string(), CovariateKind::Continuous),
        ]
    );
    assert_eq!(c.covariate_kind("SEX"), Some(CovariateKind::Categorical));
    assert_eq!(c.covariate_kind("AGE"), None);
}

#[test]
fn pk_line_parser() {
    let t = PkTemplate::parse_line("  pk one_cpt_oral( CL = CL , v=V, KA=Ka, lagtime = TLAG )")
        .expect("is a pk line")
        .expect("parses");
    assert_eq!(t.name, "one_cpt_oral");
    assert_eq!(
        t.bindings,
        vec![
            ("cl".to_string(), "CL".to_string()),
            ("v".to_string(), "V".to_string()),
            ("ka".to_string(), "Ka".to_string()),
            ("lagtime".to_string(), "TLAG".to_string()),
        ]
    );
    assert!(PkTemplate::parse_line("ode(states=[depot, central])").is_none());
    assert!(PkTemplate::parse_line("pkx(cl=CL)").is_none());
    assert!(PkTemplate::parse_line("pk one_cpt_oral cl=CL")
        .expect("is a pk line")
        .is_err());
    assert!(PkTemplate::parse_line("pk one_cpt_oral(cl)")
        .expect("is a pk line")
        .is_err());
}

#[test]
fn declared_covariate_missing_from_data_is_an_error() {
    let parsed = parse_full_model(MODEL).expect("parses");
    let text = ModelText::parse(MODEL).expect("editable");
    let e = ModelContext::from_model(&parsed, &text, &population(&["WT", "CRCL"]))
        .expect_err("SEX is not in the data");
    assert!(
        e.contains("declares `SEX` but the dataset has no such column"),
        "{e}"
    );
    assert!(e.contains("it has: WT, CRCL"), "{e}");
}

// --- one test per built-in symbol --------------------------------------------

#[test]
fn symbol_iiv() {
    assert_eq!(ctx().builtin("IIV").unwrap(), names(&["CL", "V1", "KA"]));
    let r = resolved("IIV?(@IIV,EXP)", &ctx());
    assert_eq!(r.mfl.render(), "IIV?([CL,V1,KA],EXP)");
}

#[test]
fn symbol_pk() {
    assert_eq!(
        ctx().builtin("PK").unwrap(),
        names(&["CL", "V1", "Q", "V2", "KA", "F"])
    );
    // KE is an individual parameter but not on the pk line.
    let r = resolved("IIV(@PK,EXP)", &ctx());
    assert_eq!(r.mfl.render(), "IIV([CL,V1,Q,V2,KA,F],EXP)");
}

#[test]
fn symbol_pk_iiv() {
    assert_eq!(ctx().builtin("PK_IIV").unwrap(), names(&["CL", "V1", "KA"]));
    let r = resolved("COVARIANCE(IIV,@PK_IIV)", &ctx());
    assert_eq!(r.mfl.render(), "COVARIANCE(IIV,[CL,V1,KA])");
}

#[test]
fn symbol_absorption() {
    assert_eq!(ctx().builtin("ABSORPTION").unwrap(), names(&["KA"]));
    let lag = MODEL.replace(
        "pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA, f=F)",
        "pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA, f=F, lagtime=KE)",
    );
    let c = context_for(&lag, &["WT", "SEX", "CRCL"]);
    assert_eq!(c.builtin("ABSORPTION").unwrap(), names(&["KA", "KE"]));
}

#[test]
fn symbol_elimination() {
    assert_eq!(ctx().builtin("ELIMINATION").unwrap(), names(&["CL"]));
    let r = resolved("COVARIATE?(@ELIMINATION,CRCL,pow)", &ctx());
    assert_eq!(r.mfl.render(), "COVARIATE?(CL,CRCL,pow)");
}

#[test]
fn symbol_distribution() {
    assert_eq!(
        ctx().builtin("DISTRIBUTION").unwrap(),
        names(&["V1", "Q", "V2"])
    );
}

#[test]
fn symbol_bioavail() {
    assert_eq!(ctx().builtin("BIOAVAIL").unwrap(), names(&["F"]));
    let no_f = MODEL.replace(", f=F)", ")");
    let c = context_for(&no_f, &["WT", "SEX", "CRCL"]);
    assert!(c.builtin("BIOAVAIL").unwrap().is_empty());
}

#[test]
fn symbol_continuous() {
    assert_eq!(ctx().builtin("CONTINUOUS").unwrap(), names(&["WT", "CRCL"]));
    let r = resolved("COVARIATE?(@IIV,@CONTINUOUS,[pow,lin])", &ctx());
    assert_eq!(
        r.mfl.render(),
        "COVARIATE?(CL,WT,[pow,lin]);COVARIATE?(CL,CRCL,[pow,lin]);\
         COVARIATE?(V1,WT,[pow,lin]);COVARIATE?(V1,CRCL,[pow,lin]);\
         COVARIATE?(KA,WT,[pow,lin]);COVARIATE?(KA,CRCL,[pow,lin])"
    );
    assert_eq!(r.covariate_effects.len(), 12);
}

#[test]
fn symbol_categorical() {
    assert_eq!(ctx().builtin("CATEGORICAL").unwrap(), names(&["SEX"]));
    let r = resolved("COVARIATE?(@IIV,@CATEGORICAL,cat)", &ctx());
    assert_eq!(
        r.mfl.render(),
        "COVARIATE?(CL,SEX,cat);COVARIATE?(V1,SEX,cat);COVARIATE?(KA,SEX,cat)"
    );
}

#[test]
fn symbols_pd_and_pd_iiv_are_empty_and_drop_the_feature_with_a_note() {
    assert!(ctx().builtin("PD").unwrap().is_empty());
    assert!(ctx().builtin("PD_IIV").unwrap().is_empty());
    let r = resolved("IIV(@PD,ADD);ABSORPTION(FO)", &ctx());
    assert_eq!(r.mfl.render(), "ABSORPTION(FO)");
    assert_eq!(r.notes.len(), 1);
    assert!(
        r.notes[0].contains("dropped `IIV(@PD,ADD)`"),
        "{}",
        r.notes[0]
    );
    assert!(
        r.notes[0].contains("@PD resolves to nothing"),
        "{}",
        r.notes[0]
    );
}

#[test]
fn symbol_names_are_case_insensitive_and_unknown_ones_list_the_builtins() {
    assert_eq!(ctx().builtin("iiv").unwrap(), names(&["CL", "V1", "KA"]));
    let e = resolve_err("IIV(@ETAS,EXP)", &ctx());
    assert!(e.contains("`@ETAS` is not a built-in symbol"), "{e}");
    assert!(e.contains("@CATEGORICAL"), "{e}");
}

// --- structural symbols without a pk line -----------------------------------

/// A one-compartment `[odes]` base with no template line.
const ODE_MODEL: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(40.0, 1.0, 500.0)
  omega ETA_CL ~ 0.1
  sigma ADD_ERR ~ 1.0 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[scaling]
  y = central / V

[error_model]
  DV ~ additive(ADD_ERR)
"#;

fn ode_ctx() -> ModelContext {
    context_for(ODE_MODEL, &[])
}

#[test]
fn structural_symbols_need_a_pk_line() {
    let c = ode_ctx();
    assert!(c.template.is_none());
    assert_eq!(c.builtin("IIV").unwrap(), names(&["CL"]));
    for sym in [
        "PK",
        "PK_IIV",
        "ABSORPTION",
        "ELIMINATION",
        "DISTRIBUTION",
        "BIOAVAIL",
    ] {
        let e = c.builtin(sym).expect_err(sym);
        assert!(
            e.contains("reads the `pk NAME(...)` or `ode_template NAME(...)` line"),
            "{sym}: {e}"
        );
        assert!(e.contains("LET("), "{sym}: {e}");
    }
    // LET rescues it.
    let r = resolved("LET(PK,[CL,V]);IIV(@PK,EXP)", &c);
    assert_eq!(r.mfl.render(), "IIV([CL,V],EXP)");
}

// --- covariate symbols without a [covariates] block --------------------------

#[test]
fn covariate_kind_symbols_need_a_covariates_block() {
    let undeclared = {
        let start = MODEL.find("[covariates]").unwrap();
        let end = MODEL.find("[error_model]").unwrap();
        format!("{}{}", &MODEL[..start], &MODEL[end..])
    };
    let c = context_for(&undeclared, &["WT", "SEX"]);
    assert!(c.covariates.is_none());
    let e = c.builtin("CONTINUOUS").expect_err("no block");
    assert!(
        e.contains("`@CONTINUOUS` needs a `[covariates]` block"),
        "{e}"
    );
    assert!(e.contains("continuous"), "{e}");
    let e = c.builtin("CATEGORICAL").expect_err("no block");
    assert!(e.contains("categorical"), "{e}");
    // The dataset's columns are still what an explicit name or `*` means.
    assert_eq!(c.all_covariates(), names(&["WT", "SEX"]));
    let r = resolved("COVARIATE?(CL,*,pow)", &c);
    assert_eq!(
        r.mfl.render(),
        "COVARIATE?(CL,WT,pow);COVARIATE?(CL,SEX,pow)",
        "with no kinds declared, nothing can refuse pow on SEX here"
    );
    let e = resolve_err("COVARIATE?(CL,AGE,pow)", &c);
    assert!(
        e.contains("`AGE` is not a covariate column of the dataset"),
        "{e}"
    );
    assert!(e.contains("known: WT, SEX"), "{e}");
}

// --- LET ------------------------------------------------------------------------

#[test]
fn let_defines_and_overrides_symbols() {
    let r = resolved(
        "LET(CONTINUOUS,[WT]);COVARIATE?(@IIV,@CONTINUOUS,pow)",
        &ctx(),
    );
    assert_eq!(
        r.mfl.render(),
        "COVARIATE?(CL,WT,pow);COVARIATE?(V1,WT,pow);COVARIATE?(KA,WT,pow)"
    );
    let r = resolved("LET(MINE,[CL,V2]);IIV(@mine,EXP)", &ctx());
    assert_eq!(r.mfl.render(), "IIV([CL,V2],EXP)");
    let e = resolve_err("LET(X,CL);LET(x,V1);IIV(@X,EXP)", &ctx());
    assert!(e.contains("`LET(x, ...)` is defined twice"), "{e}");
}

#[test]
fn let_values_are_validated_against_the_slot_they_land_in() {
    let e = resolve_err("LET(X,[WT]);IIV(@X,EXP)", &ctx());
    assert!(e.contains("`WT` is not an individual parameter"), "{e}");
    assert!(e.contains("it has: CL, V1, Q, V2, KA, F, KE"), "{e}");
    let e = resolve_err("COVARIATE(CL,@IIV,pow)", &ctx());
    assert!(
        e.contains("`CL` is not declared in the base model's `[covariates]` block"),
        "{e}"
    );
}

// --- wildcards ------------------------------------------------------------------

#[test]
fn wildcards_expand_by_slot() {
    let r = resolved("COVARIATE?(*,*,*)", &ctx());
    // parameters: the pk line; covariates: every declared one; effects: the
    // four continuous forms, or `cat` on a categorical covariate.
    let cl_wt: Vec<_> = r
        .covariate_effects
        .iter()
        .filter(|t| t.parameter == "CL" && t.covariate == "WT")
        .map(|t| t.effect)
        .collect();
    assert_eq!(cl_wt, CovariateEffect::CONTINUOUS.to_vec());
    let cl_sex: Vec<_> = r
        .covariate_effects
        .iter()
        .filter(|t| t.parameter == "CL" && t.covariate == "SEX")
        .map(|t| t.effect)
        .collect();
    assert_eq!(cl_sex, vec![CovariateEffect::Cat]);
    let params: std::collections::BTreeSet<_> = r
        .covariate_effects
        .iter()
        .map(|t| t.parameter.as_str())
        .collect();
    assert_eq!(params.len(), 6, "{params:?}");
    assert!(!params.contains("KE"));
    assert_eq!(r.covariate_effects.len(), 6 * (4 + 1 + 4));

    let r = resolved("IIV?(*,*)", &ctx());
    assert_eq!(
        r.mfl.render(),
        "IIV?([CL,V1,Q,V2,KA,F],[EXP,ADD,PROP,LOG,RE_LOG])"
    );
    // The parameter wildcard is feature-specific: `@PK` for IIV (any
    // parameter can get an η), `@IIV` for IOV and COVARIANCE (only an
    // η-bearing parameter can carry a κ or sit in an omega block). The
    // review caught `COVARIANCE(IIV,*)` grounding to the six PK parameters,
    // three of which have no η.
    let r = resolved("COVARIANCE(IIV,*)", &ctx());
    assert_eq!(r.mfl.render(), "COVARIANCE(IIV,[CL,V1,KA])");
    let r = resolved("IOV?(*,EXP)", &ctx());
    assert_eq!(r.mfl.render(), "IOV?([CL,V1,KA],EXP)");
    let r = resolved("COVARIANCE(*,*)", &ctx());
    assert_eq!(r.mfl.render(), "COVARIANCE([IIV,IOV],[CL,V1,KA])");
    let r = resolved("ABSORPTION(*);LAGTIME(*);TRANSITS(1,*)", &ctx());
    assert_eq!(
        r.mfl.render(),
        "ABSORPTION([INST,FO,ZO,SEQ-ZO-FO,WEIBULL]);LAGTIME([OFF,ON]);TRANSITS(1,[DEPOT,NODEPOT])"
    );
}

// --- kind checks ------------------------------------------------------------------

#[test]
fn categorical_form_on_a_continuous_covariate_and_vice_versa_are_errors() {
    let e = resolve_err("COVARIATE?(CL,SEX,pow)", &ctx());
    assert!(
        e.contains("`SEX` is declared categorical, and `pow` is a continuous form"),
        "{e}"
    );
    let e = resolve_err("COVARIATE?(CL,@CONTINUOUS,cat)", &ctx());
    assert!(
        e.contains("`WT` is declared continuous, and `cat` is a categorical form"),
        "{e}"
    );
}

// --- Pharmpy's override rules -----------------------------------------------------

#[test]
fn explicit_pair_overrides_the_symbol_derived_one() {
    let r = resolved(
        "COVARIATE?(@IIV,@CONTINUOUS,[pow,lin]);COVARIATE(CL,WT,exp)",
        &ctx(),
    );
    let cl_wt: Vec<_> = r
        .covariate_effects
        .iter()
        .filter(|t| t.parameter == "CL" && t.covariate == "WT")
        .collect();
    assert_eq!(cl_wt.len(), 1, "{cl_wt:?}");
    assert_eq!(cl_wt[0].effect, CovariateEffect::Exp);
    assert!(!cl_wt[0].optional);
    // Every other pair is untouched.
    assert_eq!(r.covariate_effects.len(), 5 * 2 + 1);
    assert!(
        r.mfl.render().contains("COVARIATE(CL,WT,exp)"),
        "{}",
        r.mfl.render()
    );
}

#[test]
fn forced_overrides_the_identical_optional_term() {
    let r = resolved("COVARIATE?(CL,WT,[pow,lin]);COVARIATE(CL,WT,pow)", &ctx());
    let mut terms: Vec<(CovariateEffect, bool)> = r
        .covariate_effects
        .iter()
        .map(|t| (t.effect, t.optional))
        .collect();
    terms.sort_by_key(|(e, o)| (e.label(), *o));
    assert_eq!(
        terms,
        vec![(CovariateEffect::Lin, true), (CovariateEffect::Pow, false)]
    );
}

#[test]
fn a_pair_forced_by_two_statements_is_an_error() {
    let e = resolve_err(
        "COVARIATE(@IIV,WT,pow);COVARIATE(@ELIMINATION,WT,exp)",
        &ctx(),
    );
    assert!(
        e.contains("CL-WT is forced by more than one COVARIATE statement"),
        "{e}"
    );
    // …but one statement forcing several effects on a pair is fine.
    resolved("COVARIATE(CL,WT,[pow,exp])", &ctx());
}

#[test]
fn covariance_needs_two_parameters() {
    let r = resolved("COVARIANCE(IIV,@ELIMINATION)", &ctx());
    assert!(r.mfl.statements.is_empty());
    assert!(
        r.notes[0].contains("resolves to 1 parameter on this model"),
        "{}",
        r.notes[0]
    );
}

#[test]
fn allometry_covariate_is_validated() {
    let r = resolved("ALLOMETRY(WT,70)", &ctx());
    assert_eq!(r.mfl.render(), "ALLOMETRY(WT,70)");
    let e = resolve_err("ALLOMETRY(BW,70)", &ctx());
    assert!(e.contains("in ALLOMETRY: `BW` is not declared"), "{e}");
}

#[test]
fn ground_output_is_stable_under_re_resolution() {
    let src = "LET(CONTINUOUS,[WT,CRCL]);ABSORPTION(*);PERIPHERALS(0..1);\
               COVARIATE?(@PK_IIV,@CONTINUOUS,*);COVARIATE?(@IIV,@CATEGORICAL,cat);\
               IIV(@PK,EXP);COVARIANCE(IIV,@PK_IIV)";
    let once = resolved(src, &ctx());
    let twice = resolve(&once.mfl, &ctx()).expect("ground MFL resolves");
    assert_eq!(once.mfl, twice.mfl);
    assert_eq!(once.covariate_effects, twice.covariate_effects);
    assert!(twice.notes.is_empty());
    assert_eq!(Mfl::parse(&once.mfl.render()).unwrap(), once.mfl);
    assert_eq!(
        CovariateEffectSpec::pair(&once.covariate_effects[0]),
        "CL-WT"
    );
}

// --- review of #1237: template, parameter and override edge cases -------------

#[test]
fn ode_template_line_is_a_template_too() {
    let t = PkTemplate::parse_line("  ode_template two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)")
        .expect("is a template line")
        .expect("parses");
    assert_eq!(t.name, "two_cpt_oral");
    assert_eq!(t.parameters(), names(&["CL", "V1", "Q", "V2", "KA"]));
    assert!(PkTemplate::parse_line("ode_templatex(cl=CL)").is_none());
    let e = PkTemplate::parse_line("ode_template two_cpt_oral cl=CL")
        .expect("is a template line")
        .expect_err("no parens");
    assert!(e.contains("`ode_template` line has no `(`"), "{e}");
}

/// A base that generates its disposition with `ode_template` binds the same
/// roles as a `pk` line, so the structural symbols must read it.
#[test]
fn structural_symbols_read_an_ode_template_base() {
    let model = MODEL.replace(
        "pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA, f=F)",
        "ode_template two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)",
    );
    let c = context_for(&model, &["WT", "SEX", "CRCL"]);
    assert_eq!(c.template.as_ref().expect("template").name, "two_cpt_oral");
    assert_eq!(
        c.builtin("PK").unwrap(),
        names(&["CL", "V1", "Q", "V2", "KA"])
    );
    assert_eq!(c.builtin("ABSORPTION").unwrap(), names(&["KA"]));
    assert_eq!(c.builtin("ELIMINATION").unwrap(), names(&["CL"]));
    assert_eq!(
        c.builtin("DISTRIBUTION").unwrap(),
        names(&["V1", "Q", "V2"])
    );
    let r = resolved("COVARIATE?(@PK,@CONTINUOUS,pow);IIV(*,EXP)", &c);
    assert_eq!(r.covariate_effects.len(), 5 * 2);
    assert!(r.notes.is_empty(), "{:?}", r.notes);
}

/// The parser lets a `pk` line bind a role to a numeric literal (`f=1`);
/// that is not an individual parameter, so the symbols must skip it rather
/// than fail on a name the user never typed.
#[test]
fn template_symbols_skip_a_literal_binding() {
    let model = MODEL
        .replace(
            "pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA, f=F)",
            "pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA, f=1)",
        )
        .replace("  F  = TVF\n", "");
    let c = context_for(&model, &["WT", "SEX", "CRCL"]);
    assert_eq!(
        c.template.as_ref().unwrap().parameters(),
        names(&["CL", "V1", "Q", "V2", "KA", "1"]),
        "the raw bindings keep the literal"
    );
    assert_eq!(
        c.builtin("PK").unwrap(),
        names(&["CL", "V1", "Q", "V2", "KA"])
    );
    assert_eq!(c.builtin("BIOAVAIL").unwrap(), Vec::<String>::new());
    // The symbol and both wildcards resolve on such a base.
    let r = resolved("COVARIATE?(@PK,@CONTINUOUS,*);IIV(*,EXP)", &c);
    assert_eq!(r.covariate_effects.len(), 5 * 2 * 4);
    let r = resolved("COVARIATE?(@BIOAVAIL,WT,pow)", &c);
    assert!(r.covariate_effects.is_empty());
    assert!(
        r.notes[0].contains("@BIOAVAIL resolves to nothing"),
        "{}",
        r.notes[0]
    );
}

/// A block-form `if` in `[individual_parameters]` is not an assignment; the
/// context reads the parser's list, not the text.
#[test]
fn block_form_if_headers_are_not_parameters() {
    let model = MODEL.replace(
        "  CL = TVCL * exp(ETA_CL)\n",
        "  if (WT >= 70) {\n    CL = TVCL * 1.2 * exp(ETA_CL)\n  } else if (SEX == 1) {\n    \
         CL = TVCL * 0.9 * exp(ETA_CL)\n  } else {\n    CL = TVCL * exp(ETA_CL)\n  }\n",
    );
    let c = context_for(&model, &["WT", "SEX", "CRCL"]);
    assert!(
        c.parameters.iter().all(|p| !p.contains('(')),
        "{:?}",
        c.parameters
    );
    assert!(
        c.parameters.contains(&"CL".to_string()),
        "{:?}",
        c.parameters
    );
    assert_eq!(c.parameters.len(), 7, "{:?}", c.parameters);
    let e = resolve_err("COVARIATE(FOO,WT,pow)", &c);
    assert!(!e.contains("if ("), "{e}");
}

/// Pharmpy gates the override on the *parameter* operand alone: a statement
/// that names its parameter explicitly is never overridden, even when its
/// covariate operand is a symbol.
#[test]
fn explicit_parameter_with_symbolic_covariate_is_not_overridden() {
    let r = resolved(
        "COVARIATE?(CL,@CONTINUOUS,[pow,lin]);COVARIATE(CL,WT,exp)",
        &ctx(),
    );
    let mut cl_wt: Vec<(CovariateEffect, bool)> = r
        .covariate_effects
        .iter()
        .filter(|t| t.parameter == "CL" && t.covariate == "WT")
        .map(|t| (t.effect, t.optional))
        .collect();
    cl_wt.sort_by_key(|(e, o)| (e.label(), *o));
    assert_eq!(
        cl_wt,
        vec![
            (CovariateEffect::Exp, false),
            (CovariateEffect::Lin, true),
            (CovariateEffect::Pow, true),
        ]
    );
    assert_eq!(r.covariate_effects.len(), 2 * 2 + 1);
    // The symbolic-parameter twin straddles the gate: its CL-WT terms go.
    let r = resolved(
        "COVARIATE?(@ELIMINATION,@CONTINUOUS,[pow,lin]);COVARIATE(CL,WT,exp)",
        &ctx(),
    );
    assert_eq!(r.covariate_effects.len(), 2 + 1);
}

/// `*` in a parameter slot is the slot's built-in symbol, so a `LET` of that
/// name governs it exactly as it governs the `@` spelling.
#[test]
fn parameter_wildcard_honours_a_let_of_its_symbol() {
    let r = resolved("LET(PK,[CL]);IIV(*,EXP)", &ctx());
    assert_eq!(r.mfl.render(), "IIV(CL,EXP)");
    let twin = resolved("LET(PK,[CL]);IIV(@PK,EXP)", &ctx());
    assert_eq!(r.mfl, twin.mfl);
    // COVARIANCE's wildcard is `@IIV`, so it is `LET(IIV, …)` that narrows it.
    let r = resolved("LET(IIV,[CL,V1]);COVARIANCE(IIV,*)", &ctx());
    assert_eq!(r.mfl.render(), "COVARIANCE(IIV,[CL,V1])");
    // On an `[odes]` base the error told the user to write `LET(PK, …)`;
    // that advice has to work for the `*` spelling too.
    let ode = context_for(ODE_MODEL, &["WT"]);
    let e = resolve_err("COVARIATE?(*,WT,pow)", &ode);
    assert!(e.contains("LET(PK, [...])"), "{e}");
    let r = resolved("LET(PK,[CL,V]);COVARIATE?(*,WT,pow)", &ode);
    assert_eq!(r.mfl.render(), "COVARIATE?(CL,WT,pow);COVARIATE?(V,WT,pow)");
}
