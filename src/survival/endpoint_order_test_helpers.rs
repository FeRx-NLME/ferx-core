//! Fixtures for the #2531 endpoint-order tests: `model.endpoints` iterates in ascending
//! CMT order on every compile, so every sum, row and message over endpoints is
//! reproducible.
//!
//! Each test **recompiles the model inside its loop**. Under a `HashMap` field every
//! compile drew a fresh `RandomState`, so recompiling in one process was enough to vary
//! the order; a single compile reuses one map and cannot fail.
//!
//! Two fixtures, because two terms added from zero commute exactly in IEEE
//! (`(0 + a) + b == (0 + b) + a`):
//! - [`EndpointOrderFixture::three_cause`] — three exponential causes (CMT 2, 3, 4) with a
//!   shared frailty. A sum of three terms from zero depends on the order; measured at
//!   `314b89fb` the FOCEI OFV took 2 distinct values (1 ULP apart) over 20 compiles.
//! - [`EndpointOrderFixture::two_cause`] — the bundled `tte_competing_risks` example. Its
//!   row order and its `S_all`/CIF bits vary with the map order, but its OFV cannot: it is
//!   the negative control of the OFV test, and needs a non-zero accumulator to be
//!   order-sensitive at all.

use crate::types::{CompiledModel, Population};
use std::io::Write;

/// Compiles per test. At `314b89fb` (a `HashMap` field) 40 compiles of the 2-cause model
/// saw both key orders (13 / 27) and of the 3-cause model all six.
pub(crate) const COMPILES: usize = 40;

/// θ, η at which the per-subject terms are evaluated: η ≠ 0 so the frailty is live.
pub(crate) const ETA: [f64; 1] = [0.3];

pub(crate) struct EndpointOrderFixture {
    src: String,
    csv_path: String,
    /// Keeps a generated CSV alive for the fixture's lifetime.
    _csv: Option<tempfile::NamedTempFile>,
}

impl EndpointOrderFixture {
    /// Three exponential causes (rates 0.10, 0.06, 0.0037) sharing one frailty η; eight
    /// subjects, each with one row per cause at its own time, the event on cause
    /// `2 + i % 4` (subject 4 and 8 censored on all three).
    pub(crate) fn three_cause() -> Self {
        let src = "[parameters]\n  theta TVA(0.10, 0.001, 10.0)\n  theta TVB(0.06, 0.001, 10.0)\n  \
                   theta TVC(0.0037, 0.0001, 10.0)\n  omega ETA_F ~ 0.25\n\
                   [event_model cause_a]\n  cmt = 2\n  family = exponential\n  scale = TVA * exp(ETA_F)\n\
                   [event_model cause_b]\n  cmt = 3\n  family = exponential\n  scale = TVB * exp(ETA_F)\n\
                   [event_model cause_c]\n  cmt = 4\n  family = exponential\n  scale = TVC * exp(ETA_F)\n"
            .to_string();
        let mut csv = String::from("ID,TIME,DV,EVID,AMT,CMT,MDV\n");
        let times = [12.2538, 9.377, 2.7931, 5.6074, 14.0, 3.3, 7.77, 1.234];
        for (i, t) in times.iter().enumerate() {
            let event_cmt = 2 + (i % 4); // 5 => censored on every cause
            for cmt in 2..=4 {
                let dv = usize::from(cmt == event_cmt);
                csv.push_str(&format!("{},{t},{dv},0,0,{cmt},0\n", i + 1));
            }
        }
        let mut f = tempfile::NamedTempFile::new().expect("temp csv");
        f.write_all(csv.as_bytes()).expect("write csv");
        Self {
            src,
            csv_path: f.path().to_str().expect("utf-8 path").to_string(),
            _csv: Some(f),
        }
    }

    /// The bundled two-cause competing-risks example (CMT 2, 3).
    pub(crate) fn two_cause() -> Self {
        let dir = env!("CARGO_MANIFEST_DIR");
        Self {
            src: std::fs::read_to_string(format!("{dir}/examples/tte_competing_risks.ferx"))
                .expect("example"),
            csv_path: format!("{dir}/data/tte_competing_risks.csv"),
            _csv: None,
        }
    }

    /// A fresh compile of the model, and its data read through the model-aware reader.
    pub(crate) fn compile(&self) -> (CompiledModel, Population) {
        let m = crate::parser::model_parser::parse_model_string(&self.src).expect("parse");
        let pop = crate::api::read_population_for(&m, &None, &self.csv_path, None, None, None, &[])
            .expect("data must load")
            .0;
        (m, pop)
    }

    /// A fresh compile keeping only the endpoint on `cmt`: the term that endpoint alone
    /// contributes, through the same code path as the full model.
    pub(crate) fn compile_only(&self, cmt: usize) -> CompiledModel {
        let mut m = self.compile().0;
        m.endpoints.retain(|&k, _| k == cmt);
        assert_eq!(m.endpoints.len(), 1, "fixture has no endpoint on CMT {cmt}");
        m
    }

    /// The fixture's endpoint CMTs, ascending.
    pub(crate) fn cmts(&self) -> Vec<usize> {
        let mut c: Vec<usize> = self.compile().0.endpoints.keys().copied().collect();
        c.sort_unstable();
        c
    }
}

/// `acc0` plus `terms` added left to right — the sum in the order given.
pub(crate) fn fold(acc0: f64, terms: &[f64]) -> f64 {
    terms.iter().fold(acc0, |acc, &t| acc + t)
}

/// Whether adding `terms` to `acc0` in some order gives other bits than ascending order:
/// the straddle a differential test needs, or the order-pin cannot fail.
pub(crate) fn order_sensitive(acc0: f64, terms: &[f64]) -> bool {
    let want = fold(acc0, terms).to_bits();
    let mut idx: Vec<usize> = (0..terms.len()).collect();
    permutations(&mut idx, 0)
        .iter()
        .any(|p| fold(acc0, &p.iter().map(|&i| terms[i]).collect::<Vec<_>>()).to_bits() != want)
}

fn permutations(idx: &mut Vec<usize>, k: usize) -> Vec<Vec<usize>> {
    if k == idx.len() {
        return vec![idx.clone()];
    }
    let mut out = Vec::new();
    for i in k..idx.len() {
        idx.swap(k, i);
        out.extend(permutations(idx, k + 1));
        idx.swap(k, i);
    }
    out
}
