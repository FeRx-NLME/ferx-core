//! The Pharmpy anchor for the MFL parser and the `@`-symbol resolver.
//!
//! `tests/data/mfl_pharmpy_anchor.json` is Pharmpy's own reading of every
//! line of `tools/pharmpy-mfl-anchor/corpus.txt` (raw `parse()`, no model)
//! plus its `expand()` of a set of symbol-bearing statements on a reference
//! model. This module parses / resolves the same strings with ferx and
//! compares. Regenerate the JSON with `tools/pharmpy-mfl-anchor/run.sh`.
//!
//! Where ferx deliberately differs, the line is named in
//! [`PARSE_DIVERGENCES`] / [`EXPAND_DIVERGENCES`] with the reason, and the
//! test asserts the two sides really *do* differ — so a divergence that
//! quietly disappears (or a new one) is a red test either way.

use std::collections::BTreeSet;

use ferx_core::edit::ModelText;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{Population, Subject};
use serde_json::{json, Value};

use super::super::resolve::{resolve, ModelContext};
use super::*;

const ANCHOR: &str = include_str!("../../tests/data/mfl_pharmpy_anchor.json");

/// Corpus lines where ferx and Pharmpy 2.0.0 disagree on purpose, with why.
/// Each must actually differ (checked); see tools/pharmpy-mfl-anchor/README.md.
const PARSE_DIVERGENCES: &[(&str, &str)] = &[
    (
        "PERIPHERALS(2..0)",
        "Pharmpy yields an empty count set silently; ferx refuses a backwards range",
    ),
    (
        "PERIPHERALS(0..100)",
        "Pharmpy materialises any range; ferx bounds a count at MAX_COUNT (64) so a typo cannot \
         allocate a 16 GiB vector at config-load time",
    ),
    (
        "ALLOMETRY(WT)",
        "Pharmpy raises IndexError (a bug: its grammar documents the reference as optional)",
    ),
    (
        "COVARIATE?(*,*,*)",
        "Pharmpy's interpreter crashes (TypeError) on the parameter/covariate wildcards its grammar allows",
    ),
    (
        "COVARIATE?(CL,*,pow)",
        "Pharmpy's interpreter crashes (TypeError) on the covariate wildcard its grammar allows",
    ),
    (
        "COVARIATE?(*,WT,pow)",
        "Pharmpy's interpreter crashes (TypeError) on the parameter wildcard its grammar allows",
    ),
    (
        "ABSORPTION(FO) # c",
        "ferx accepts `#` comments inside the TOML string; Pharmpy has no comment syntax",
    ),
    (
        "ABSORPTION(FO);;\nELIMINATION(FO)",
        "ferx tolerates empty statements between separators; Pharmpy does not",
    ),
    ("IIV(@PK,EXP)", "IIV is in Pharmpy's docs, not in its 2.0.0 grammar"),
    ("IIV?([CL,V],[EXP,ADD])", "IIV is in Pharmpy's docs, not in its 2.0.0 grammar"),
    ("IOV(CL,PROP)", "IOV is in Pharmpy's docs, not in its 2.0.0 grammar"),
    (
        "COVARIANCE(IIV,@PK_IIV)",
        "COVARIANCE is in Pharmpy's docs, not in its 2.0.0 grammar",
    ),
    (
        "COVARIANCE?(IOV,[CL,V])",
        "COVARIANCE is in Pharmpy's docs, not in its 2.0.0 grammar",
    ),
];

/// Expand cases where the resolved covariate sets differ on purpose.
const EXPAND_DIVERGENCES: &[(&str, &str)] = &[
    (
        "COVARIATE?(@PD,WT,pow)",
        "Pharmpy raises ValueError with no PD parameters; ferx drops the statement with a note",
    ),
    (
        "COVARIATE?(@NOSUCH,WT,pow)",
        "Pharmpy resolves an unknown symbol to () silently; ferx errors naming it",
    ),
    (
        "COVARIATE?(@ABSORPTION,WT,pow);COVARIATE?(@DISTRIBUTION,WT,cat)",
        "Pharmpy never checks a covariate's kind against the effect; ferx refuses `cat` on the \
         continuous `WT` at resolution rather than in fifty candidates' parsers",
    ),
    (
        "COVARIATE?(@PK,@CONTINUOUS,[pow,lin]);COVARIATE(CL,WT,exp)",
        "an explicit pair makes Pharmpy drop the whole parameter from the symbol statement \
         (CL-CRCL vanishes); ferx drops only the pair",
    ),
    (
        "COVARIATE?(@IIV,[WT,CRCL],*);COVARIATE?(CL,WT,exp)",
        "an explicit pair makes Pharmpy drop the whole parameter from the symbol statement \
         (CL-CRCL vanishes); ferx drops only the pair",
    ),
];

fn anchor() -> Value {
    serde_json::from_str(ANCHOR).expect("anchor JSON parses")
}

// --- ferx → Pharmpy's JSON shape ------------------------------------------------

fn upper(names: &[String]) -> Value {
    Value::Array(
        names
            .iter()
            .map(|n| json!(n.to_ascii_uppercase()))
            .collect(),
    )
}

fn modes_json<M: Mode>(m: &Modes<M>) -> Value {
    match m {
        Modes::Wildcard => json!("*"),
        Modes::List(list) => Value::Array(
            list.iter()
                .map(|m| json!(m.label().to_ascii_uppercase()))
                .collect(),
        ),
    }
}

fn operand_json(o: &Operand) -> Value {
    match o {
        Operand::Wildcard => json!("*"),
        Operand::Symbol(s) => json!(format!("@{s}")),
        Operand::Names(names) => upper(names),
    }
}

/// Counts as Pharmpy holds them: a range expanded, a list as written.
fn counts_json(c: &Counts) -> Value {
    let v: Vec<u32> = match c {
        Counts::Single(n) => vec![*n],
        Counts::Range(a, b) => (*a..=*b).collect(),
        Counts::List(l) => l.clone(),
    };
    json!(v)
}

/// The ferx statement in Pharmpy's serialised shape. The three documented
/// deviations are applied here and nowhere else: names upper-cased, and the
/// defaults Pharmpy fills at parse time (`DRUG`, `DEPOT` / `NODEPOT`) filled
/// where ferx keeps "unspecified".
fn ferx_json(s: &Statement) -> Value {
    match s {
        Statement::Let { name, values } => {
            json!({"type": "LET", "name": name, "value": upper(values)})
        }
        Statement::Feature(f) => match f {
            Feature::Absorption(m) => json!({"type": "ABSORPTION", "modes": modes_json(m)}),
            Feature::Elimination(m) => json!({"type": "ELIMINATION", "modes": modes_json(m)}),
            Feature::Peripherals { counts, kind } => json!({
                "type": "PERIPHERALS",
                "counts": counts_json(counts),
                "modes": kind.as_ref().map(modes_json).unwrap_or_else(|| json!(["DRUG"])),
            }),
            Feature::Transits { counts, depot } => {
                let (counts, default_depot) = match counts {
                    TransitCounts::N => (json!(["N"]), json!(["NODEPOT"])),
                    TransitCounts::Counts(c) => (counts_json(c), json!(["DEPOT"])),
                };
                json!({
                    "type": "TRANSITS",
                    "counts": counts,
                    "depot": depot.as_ref().map(modes_json).unwrap_or(default_depot),
                })
            }
            Feature::Lagtime(m) => json!({"type": "LAGTIME", "modes": modes_json(m)}),
            Feature::Covariate {
                optional,
                parameters,
                covariates,
                effects,
                op,
            } => json!({
                "type": "COVARIATE",
                "parameter": operand_json(parameters),
                "covariate": operand_json(covariates),
                "fp": modes_json(effects),
                "op": op.label(),
                "optional": optional,
            }),
            Feature::Allometry {
                covariate,
                reference,
            } => json!({
                "type": "ALLOMETRY",
                "covariate": covariate.to_ascii_uppercase(),
                "reference": reference.as_ref().map(|r| r.parse::<f64>().expect("numeric")),
            }),
            Feature::DirectEffect(m) => json!({"type": "DIRECTEFFECT", "modes": modes_json(m)}),
            Feature::EffectComp(m) => json!({"type": "EFFECTCOMP", "modes": modes_json(m)}),
            Feature::IndirectEffect { mode, production } => json!({
                "type": "INDIRECTEFFECT",
                "modes": modes_json(mode),
                "production": modes_json(production),
            }),
            Feature::Metabolite(m) => json!({"type": "METABOLITE", "modes": modes_json(m)}),
            Feature::Iiv { .. } | Feature::Iov { .. } | Feature::Covariance { .. } => {
                json!({"type": f.keyword(), "not_in_pharmpy": true})
            }
        },
    }
}

/// Pharmpy keeps a repeated mode (`LAGTIME([ON,ON,OFF])`); ferx collapses it.
/// Compare after collapsing Pharmpy's side too — a *set* of modes is what
/// either grammar means.
fn dedup_modes(v: &Value) -> Value {
    match v {
        Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (k, v) in map {
                let v = if matches!(k.as_str(), "modes" | "depot" | "fp" | "production") {
                    match v {
                        Value::Array(items) => {
                            let mut seen = Vec::new();
                            for item in items {
                                if !seen.contains(item) {
                                    seen.push(item.clone());
                                }
                            }
                            Value::Array(seen)
                        }
                        other => other.clone(),
                    }
                } else {
                    v.clone()
                };
                out.insert(k.clone(), v);
            }
            Value::Object(out)
        }
        other => other.clone(),
    }
}

#[test]
fn every_corpus_line_agrees_with_pharmpy_or_is_a_named_divergence() {
    let doc = anchor();
    assert_eq!(
        doc["pharmpy_version"], "2.0.0",
        "regenerate: the divergence list is per version"
    );
    let entries = doc["parse"].as_array().expect("parse array");
    assert!(entries.len() >= 70, "corpus shrank to {}", entries.len());

    let mut divergences_seen: BTreeSet<&str> = BTreeSet::new();
    for entry in entries {
        let mfl = entry["mfl"].as_str().expect("mfl");
        let pharmpy: Result<Vec<Value>, String> = match entry.get("statements") {
            Some(Value::Array(s)) => Ok(s.iter().map(dedup_modes).collect()),
            _ => Err(entry["error"].as_str().unwrap_or("?").to_string()),
        };
        let ferx: Result<Vec<Value>, String> =
            Mfl::parse(mfl).map(|m| m.statements.iter().map(ferx_json).collect());

        if let Some((_, why)) = PARSE_DIVERGENCES.iter().find(|(line, _)| *line == mfl) {
            divergences_seen.insert(mfl);
            assert_ne!(
                ferx, pharmpy,
                "{mfl:?} is listed as a divergence ({why}) but the two sides now agree — \
                 drop it from PARSE_DIVERGENCES"
            );
            continue;
        }
        match (&ferx, &pharmpy) {
            (Ok(f), Ok(p)) => assert_eq!(
                f, p,
                "{mfl:?}: ferx and Pharmpy parse differently\n ferx:    {f:?}\n pharmpy: {p:?}"
            ),
            (Err(_), Err(_)) => {}
            (Ok(f), Err(e)) => panic!("{mfl:?}: Pharmpy rejects it ({e}) but ferx parses {f:?}"),
            (Err(e), Ok(p)) => panic!("{mfl:?}: ferx rejects it ({e}) but Pharmpy parses {p:?}"),
        }
    }
    let listed: BTreeSet<&str> = PARSE_DIVERGENCES.iter().map(|(l, _)| *l).collect();
    assert_eq!(
        divergences_seen, listed,
        "every listed divergence must be in the corpus (and vice versa)"
    );
}

// --- expand() ----------------------------------------------------------------------

/// The ferx twin of `dump.py`'s reference model: Pharmpy's basic oral model
/// with one peripheral compartment, its parameter names, η on CL / VC / MAT,
/// and the three covariates.
const TWIN: &str = r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVVC(40.0, 1.0, 500.0)
  theta TVQP1(8.0, 0.1, 100.0)
  theta TVVP1(80.0, 1.0, 500.0)
  theta TVMAT(1.0, 0.01, 10.0)
  omega ETA_CL ~ 0.1
  omega ETA_VC ~ 0.1
  omega ETA_MAT ~ 0.1
  sigma PROP_ERR ~ 0.04 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  VC  = TVVC * exp(ETA_VC)
  QP1 = TVQP1
  VP1 = TVVP1
  MAT = TVMAT * exp(ETA_MAT)

[structural_model]
  pk two_cpt_oral(cl=CL, v1=VC, q=QP1, v2=VP1, ka=MAT)

[covariates]
  WT   continuous
  CRCL continuous
  SEX  categorical(levels = [0, 1])

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

fn twin_context() -> ModelContext {
    let parsed = parse_full_model(TWIN).expect("twin parses");
    let text = ModelText::parse(TWIN).expect("twin is editable");
    let population = Population {
        subjects: vec![Subject {
            id: "1".into(),
            ..Default::default()
        }],
        covariate_names: vec!["WT".into(), "CRCL".into(), "SEX".into()],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
    ModelContext::from_model(&parsed, &text, &population).expect("context")
}

type Tuple = (String, String, String, String, bool);

fn set_of(v: &Value) -> BTreeSet<Tuple> {
    v.as_array()
        .expect("covariates array")
        .iter()
        .map(|t| {
            let t = t.as_array().expect("tuple");
            (
                t[0].as_str().unwrap().to_string(),
                t[1].as_str().unwrap().to_string(),
                t[2].as_str().unwrap().to_string(),
                t[3].as_str().unwrap().to_string(),
                t[4].as_bool().unwrap(),
            )
        })
        .collect()
}

fn names(v: &Value) -> BTreeSet<String> {
    v.as_array()
        .expect("names")
        .iter()
        .map(|s| s.as_str().unwrap().to_string())
        .collect()
}

#[test]
fn the_builtin_symbols_resolve_to_pharmpys_sets_on_the_twin_model() {
    let model = &anchor()["expand"]["model"];
    let ctx = twin_context();
    let as_set = |v: Vec<String>| v.into_iter().collect::<BTreeSet<_>>();
    for (symbol, key) in [
        ("PK", "pk"),
        ("IIV", "iiv"),
        ("ABSORPTION", "absorption"),
        ("ELIMINATION", "elimination"),
        ("DISTRIBUTION", "distribution"),
        ("CONTINUOUS", "continuous"),
        ("CATEGORICAL", "categorical"),
    ] {
        assert_eq!(
            as_set(ctx.builtin(symbol).unwrap()),
            names(&model[key]),
            "@{symbol}"
        );
    }
    assert!(
        !names(&model["categorical"]).is_empty(),
        "the reference model must carry a categorical covariate or @CATEGORICAL is untested"
    );
}

#[test]
fn expand_cases_agree_with_pharmpy_or_are_named_divergences() {
    let doc = anchor();
    let ctx = twin_context();
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for case in doc["expand"]["cases"].as_array().expect("cases") {
        let mfl = case["mfl"].as_str().expect("mfl");
        let pharmpy: Result<BTreeSet<Tuple>, String> = match case.get("covariates") {
            Some(v) => Ok(set_of(v)),
            None => Err(case["error"].as_str().unwrap_or("?").to_string()),
        };
        let ferx: Result<BTreeSet<Tuple>, String> =
            resolve(&Mfl::parse(mfl).expect("parses"), &ctx).map(|r| {
                r.covariate_effects
                    .iter()
                    .map(|t| {
                        (
                            t.parameter.clone(),
                            t.covariate.clone(),
                            t.effect.label().to_ascii_uppercase(),
                            t.op.label().to_string(),
                            t.optional,
                        )
                    })
                    .collect()
            });

        if let Some((_, why)) = EXPAND_DIVERGENCES.iter().find(|(line, _)| *line == mfl) {
            seen.insert(mfl);
            assert_ne!(
                ferx, pharmpy,
                "{mfl:?}: listed divergence ({why}) no longer differs"
            );
            if let (Ok(f), Ok(p)) = (&ferx, &pharmpy) {
                // The override divergence is exactly the CL-CRCL tuples Pharmpy
                // lost: ferx ⊃ Pharmpy, and the surplus is all `CL` on a
                // covariate other than the explicit `WT`.
                assert!(
                    p.is_subset(f),
                    "{mfl:?}: ferx must keep everything Pharmpy keeps"
                );
                let surplus: Vec<&Tuple> = f.difference(p).collect();
                assert!(!surplus.is_empty());
                assert!(
                    surplus.iter().all(|t| t.0 == "CL" && t.1 != "WT"),
                    "{mfl:?}: unexpected surplus {surplus:?}"
                );
            }
            continue;
        }
        match (&ferx, &pharmpy) {
            (Ok(f), Ok(p)) => assert_eq!(f, p, "{mfl:?}: resolved sets differ"),
            (Err(_), Err(_)) => {}
            (Ok(f), Err(e)) => panic!("{mfl:?}: Pharmpy fails ({e}) but ferx resolves {f:?}"),
            (Err(e), Ok(p)) => panic!("{mfl:?}: ferx fails ({e}) but Pharmpy resolves {p:?}"),
        }
    }
    let listed: BTreeSet<&str> = EXPAND_DIVERGENCES.iter().map(|(l, _)| *l).collect();
    assert_eq!(
        seen, listed,
        "every listed divergence must be an expand case (and vice versa)"
    );
}
