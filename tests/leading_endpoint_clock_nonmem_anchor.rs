//! NONMEM anchor for #3095: an endpoint row ahead of every other record of a subject with
//! PK records is a record, and starts the system there.
//!
//! NONMEM has no endpoint kinds, so a count row is a record like any other, and the initial
//! state applies at the subject's first record. `nonmem_anchor/leading_endpoint_clock.ferx`
//! seeds a central baseline (`init(central) = 200`) that decays from that first record, so
//! the state moves before the first dose.
//! NONMEM's twin is `ADVAN2` with `A_0(2)`. In `leading_endpoint_clock.csv`, IDs 1–3 open
//! with a count row (`LEAD = 1`) before their first dose. NONMEM reads that row as an
//! `EVID=2` record (`leading_endpoint_clock_nm.csv`):
//!
//! * arm a keeps it, so the system starts at the count row's TIME;
//! * arm b ignores it (`IGNORE=(LEAD.EQ.1)`), so the system starts at the dose. This is the
//!   straddle, and what ferx did before #3095, when the leading row started no clock.
//!
//! IDs 4 and 5 are controls that read the same under either rule: a count row after the
//! first dose, and one tied with it in file order ahead of it.
//!
//! Tier 2: one `predict()` at the model file's fixed values.

#![cfg(feature = "survival")]

use ferx_core::api::read_population_for;
use ferx_core::{parse_full_model_file, predict};
use std::path::Path;

/// `(ID, TIME, IPRED)` of every observation (EVID = 0) row in `nonmem_anchor/results/<name>.tab`.
fn nonmem_ipred(name: &str) -> Vec<(i64, f64, f64)> {
    let path = format!("nonmem_anchor/results/{name}.tab");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut lines = text.lines().skip_while(|l| !l.trim_start().starts_with("ID"));
    let header: Vec<&str> = lines.next().expect("header").split_whitespace().collect();
    let col = |c: &str| header.iter().position(|h| *h == c).expect(c);
    let (id, time, evid, ipred) = (col("ID"), col("TIME"), col("EVID"), col("IPRED"));
    let rows = lines.map(|l| l.split_whitespace().map(|v| v.parse::<f64>().unwrap()).collect());
    rows.filter(|r: &Vec<f64>| r[evid] == 0.0).map(|r| (r[id] as i64, r[time], r[ipred])).collect()
}

#[test]
fn a_leading_count_row_starts_the_system_as_nonmem_s_record_does() {
    let (a, b) =
        (nonmem_ipred("leading_endpoint_clock_a"), nonmem_ipred("leading_endpoint_clock_b"));
    let m = parse_full_model_file(Path::new("nonmem_anchor/leading_endpoint_clock.ferx")).unwrap();
    let csv = "nonmem_anchor/leading_endpoint_clock.csv";
    let pop = read_population_for(&m.model, &None, csv, None, None, None, &[]).unwrap().0;
    let got = predict(&m.model, &pop, &m.model.default_params).unwrap();
    assert_eq!((got.len(), a.len(), b.len()), (19, 19, 19), "observation rows");
    for ((p, &(id, t, want)), &(_, _, ctrl)) in got.iter().zip(&a).zip(&b) {
        assert!(p.id == id.to_string() && (p.time - t).abs() < 1e-9, "row ({id}, {t})");
        // The straddle: the leading record moves IDs 1–3 by far more than the bound, and the
        // controls not at all.
        assert_eq!(id <= 3, (want - ctrl).abs() > 1e-3, "ID {id} TIME {t}: {want} / {ctrl}");
        let err = (p.pred - want).abs();
        assert!(err < 1e-6, "ID {id} TIME {t}: ferx {} vs NONMEM {want} ({err:.3e})", p.pred);
    }
}
