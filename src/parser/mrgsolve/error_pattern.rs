//! Pattern-match a preprocessed `$TABLE` / `$ERROR` body against ferx's
//! three supported residual-error shapes (proportional, additive, combined).
//!
//! ferx represents residual error declaratively — `DV ~ proportional(SIGMA)`,
//! `DV ~ additive(SIGMA)`, `DV ~ combined(PROP, ADD)`. mrgsolve's `$TABLE`
//! body is free-form C++, but the *terminal* expression that defines the
//! observation typically has one of three structural shapes. We recognise
//! those shapes via regex on the preprocessed text and reject anything else
//! with a clear error.

use super::name_map::NameMap;
use std::sync::LazyLock;

use regex::Regex;

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ClassifiedError {
    Proportional { sigma: String },
    Additive { sigma: String },
    Combined { sigma_prop: String, sigma_add: String },
}

/// Inspect a preprocessed `$TABLE` / `$ERROR` body and return the matched
/// ferx error model. Returns `Err` with the supported-patterns hint if no
/// pattern matches.
pub(crate) fn classify(body: &str, name_map: &NameMap) -> Result<ClassifiedError, String> {
    // Walk the body in reverse — the observation assignment is usually the
    // last EPS-bearing statement. The first match wins.
    let mut last_eps_line: Option<&str> = None;
    for line in body.lines().rev() {
        if INDEXED_EPS.is_match(line) {
            last_eps_line = Some(line.trim());
            break;
        }
    }
    let line = last_eps_line.ok_or_else(|| {
        format!(
            "no `EPS(...)` reference found in $TABLE/$ERROR body — cannot infer error model"
        )
    })?;

    // Try each pattern in order of specificity. Combined first so it
    // doesn't get classified as Proportional by mistake.
    if let Some(c) = CMB.captures(line) {
        let p_idx: usize = c[1].parse().unwrap();
        let a_idx: usize = c[2].parse().unwrap();
        return Ok(ClassifiedError::Combined {
            sigma_prop: name_map.eps(p_idx).map_err(prefix)?.to_string(),
            sigma_add: name_map.eps(a_idx).map_err(prefix)?.to_string(),
        });
    }
    if let Some(c) = PROP.captures(line) {
        let p_idx: usize = c[1].parse().unwrap();
        return Ok(ClassifiedError::Proportional {
            sigma: name_map.eps(p_idx).map_err(prefix)?.to_string(),
        });
    }
    if let Some(c) = ADD.captures(line) {
        let a_idx: usize = c[1].parse().unwrap();
        return Ok(ClassifiedError::Additive {
            sigma: name_map.eps(a_idx).map_err(prefix)?.to_string(),
        });
    }

    Err(format!(
        "could not classify error model from line: `{}`\n\
         ferx supports: `Y = IPRED * (1 + EPS(p))` (proportional), \
         `Y = IPRED + EPS(a)` (additive), \
         or `Y = IPRED * (1 + EPS(p)) + EPS(a)` (combined). \
         Rewrite the $TABLE/$ERROR assignment to one of these forms.",
        line
    ))
}

fn prefix(s: String) -> String {
    format!("$TABLE: {s}")
}

// ── Regex catalogue ────────────────────────────────────────────────────────

static INDEXED_EPS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\bEPS\s*\(\s*\d+\s*\)").unwrap());

// `Y = <anything> * (1 + EPS(p)) + EPS(a)` — combined.
static CMB: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?xi)
            =                            # assignment
            \s* .+? \s*                  # IPRED expression (any)
            \*                           # multiply
            \s* \( \s* 1 \s* \+ \s*
                EPS \s* \( \s* (\d+) \s* \)
            \s* \)                       # close (1+EPS(p))
            \s* \+ \s*
            EPS \s* \( \s* (\d+) \s* \)
            \s* $                        # end of statement
        ",
    )
    .unwrap()
});

// `Y = <anything> * (1 + EPS(p))` — proportional.
static PROP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?xi)
            =
            \s* .+? \s*
            \*
            \s* \( \s* 1 \s* \+ \s*
                EPS \s* \( \s* (\d+) \s* \)
            \s* \)
            \s* $
        ",
    )
    .unwrap()
});

// `Y = <anything> + EPS(a)` — additive.
static ADD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?xi)
            =
            \s* .+? \s*
            \+ \s*
            EPS \s* \( \s* (\d+) \s* \)
            \s* $
        ",
    )
    .unwrap()
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::mrgsolve::intermediate::{MrgModel, OmegaBlock};

    fn nm(sigma_names: &[&str]) -> NameMap {
        let m = MrgModel {
            sigmas: vec![OmegaBlock {
                block_form: false,
                labels: sigma_names.iter().map(|s| s.to_string()).collect(),
                values: vec![0.01; sigma_names.len()],
            }],
            ..MrgModel::default()
        };
        NameMap::from_model(&m)
    }

    #[test]
    fn classifies_proportional() {
        let body = "IPRED = CENT/V\nY = IPRED * (1 + EPS(1))\n";
        let nm = nm(&["PROP"]);
        match classify(body, &nm).unwrap() {
            ClassifiedError::Proportional { sigma } => assert_eq!(sigma, "PROP"),
            other => panic!("expected proportional, got {:?}", other),
        }
    }

    #[test]
    fn classifies_additive() {
        let body = "Y = IPRED + EPS(1)\n";
        let nm = nm(&["ADD_ERR"]);
        match classify(body, &nm).unwrap() {
            ClassifiedError::Additive { sigma } => assert_eq!(sigma, "ADD_ERR"),
            other => panic!("expected additive, got {:?}", other),
        }
    }

    #[test]
    fn classifies_combined() {
        let body = "Y = IPRED * (1 + EPS(1)) + EPS(2)\n";
        let nm = nm(&["PROP", "ADD"]);
        match classify(body, &nm).unwrap() {
            ClassifiedError::Combined {
                sigma_prop,
                sigma_add,
            } => {
                assert_eq!(sigma_prop, "PROP");
                assert_eq!(sigma_add, "ADD");
            }
            other => panic!("expected combined, got {:?}", other),
        }
    }

    #[test]
    fn rejects_lognormal_form() {
        let body = "Y = IPRED * exp(EPS(1))\n";
        let nm = nm(&["LOG_ERR"]);
        let err = classify(body, &nm).unwrap_err();
        assert!(err.contains("could not classify"));
        assert!(err.contains("ferx supports"));
    }

    #[test]
    fn rejects_no_eps() {
        let body = "Y = IPRED\n";
        let nm = nm(&["SIGMA"]);
        let err = classify(body, &nm).unwrap_err();
        assert!(err.contains("no `EPS(...)` reference"));
    }

    #[test]
    fn out_of_range_eps_propagates() {
        let body = "Y = IPRED * (1 + EPS(7))\n";
        let nm = nm(&["PROP"]);
        let err = classify(body, &nm).unwrap_err();
        assert!(err.contains("out of range"));
    }
}
