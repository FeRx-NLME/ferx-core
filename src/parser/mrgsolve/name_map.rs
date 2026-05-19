//! Resolve `THETA(n)` / `ETA(n)` / `EPS(n)` 1-indexed access to ferx-named
//! parameters.
//!
//! mrgsolve lets users mix positional access (`THETA(1)`) with named access
//! (`TVCL`) inside the same model. ferx parameters are always named, so the
//! preprocessor needs a way to look up "the Nth theta" → "TVCL". This module
//! owns the tables that link the two worlds.

use super::intermediate::{MrgModel, OmegaBlock};

/// Lookup tables built from a parsed `MrgModel`. Pass into the C++ body
/// preprocessor to resolve `THETA(n)`/`ETA(n)`/`EPS(n)` references.
#[derive(Debug, Clone)]
pub(crate) struct NameMap {
    pub(crate) theta_names: Vec<String>,
    pub(crate) eta_names: Vec<String>,
    pub(crate) eps_names: Vec<String>,
}

impl NameMap {
    /// Build a name map from an MrgModel by flattening omega/sigma blocks.
    pub(crate) fn from_model(m: &MrgModel) -> Self {
        Self {
            theta_names: m.thetas.iter().map(|t| t.name.clone()).collect(),
            eta_names: flatten_labels(&m.omegas),
            eps_names: flatten_labels(&m.sigmas),
        }
    }

    /// 1-indexed lookup. Returns `Err` with the supported range if `n` is
    /// out of bounds.
    pub(crate) fn theta(&self, n: usize) -> Result<&str, String> {
        lookup_1based("THETA", &self.theta_names, n)
    }

    pub(crate) fn eta(&self, n: usize) -> Result<&str, String> {
        lookup_1based("ETA", &self.eta_names, n)
    }

    pub(crate) fn eps(&self, n: usize) -> Result<&str, String> {
        lookup_1based("EPS", &self.eps_names, n)
    }
}

fn flatten_labels(blocks: &[OmegaBlock]) -> Vec<String> {
    blocks.iter().flat_map(|b| b.labels.iter().cloned()).collect()
}

fn lookup_1based<'a>(kind: &str, names: &'a [String], n: usize) -> Result<&'a str, String> {
    if n == 0 {
        return Err(format!("{kind}(0) is invalid (1-indexed)"));
    }
    if n > names.len() {
        return Err(format!(
            "{kind}({n}) is out of range (only {} {kind} declared)",
            names.len()
        ));
    }
    Ok(names[n - 1].as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::mrgsolve::intermediate::{NamedValue, OmegaBlock};

    fn make_model() -> MrgModel {
        MrgModel {
            thetas: vec![
                NamedValue {
                    name: "TVCL".into(),
                    value: 1.0,
                },
                NamedValue {
                    name: "TVV".into(),
                    value: 10.0,
                },
            ],
            omegas: vec![OmegaBlock {
                block_form: false,
                labels: vec!["ETA_CL".into(), "ETA_V".into()],
                values: vec![0.09, 0.04],
            }],
            sigmas: vec![OmegaBlock {
                block_form: false,
                labels: vec!["PROP_ERR".into()],
                values: vec![0.02],
            }],
            ..MrgModel::default()
        }
    }

    #[test]
    fn resolves_in_range_indices() {
        let m = make_model();
        let nm = NameMap::from_model(&m);
        assert_eq!(nm.theta(1).unwrap(), "TVCL");
        assert_eq!(nm.theta(2).unwrap(), "TVV");
        assert_eq!(nm.eta(1).unwrap(), "ETA_CL");
        assert_eq!(nm.eta(2).unwrap(), "ETA_V");
        assert_eq!(nm.eps(1).unwrap(), "PROP_ERR");
    }

    #[test]
    fn rejects_zero_and_out_of_range() {
        let m = make_model();
        let nm = NameMap::from_model(&m);
        assert!(nm.theta(0).unwrap_err().contains("1-indexed"));
        assert!(nm.theta(3).unwrap_err().contains("out of range"));
        assert!(nm.eta(3).unwrap_err().contains("out of range"));
    }

    #[test]
    fn flattens_multi_block_omega() {
        let mut m = make_model();
        m.omegas.push(OmegaBlock {
            block_form: true,
            labels: vec!["ETA_KA".into(), "ETA_F".into()],
            values: vec![0.3, 0.1, 0.2],
        });
        let nm = NameMap::from_model(&m);
        assert_eq!(nm.eta_names, vec!["ETA_CL", "ETA_V", "ETA_KA", "ETA_F"]);
    }
}
