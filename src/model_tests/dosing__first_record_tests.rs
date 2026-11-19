// --- the imports of the original parent module (relocated test, #3041) ---
use ferx_model::types::DoseEvent;
use std::borrow::Cow;

use crate::types::Subject;
use ferx_model::dosing::{first_record, FirstRecord};
use std::io::Write;

/// Each subject is a tie at TIME 0 between two records of the kinds in its name, in
/// the file order its name gives, followed by an observation at TIME 2. The first row
/// of every tie carries WT = 70 and the second WT = 140, so the first record in file
/// order — NONMEM's `A_0` seed (`nonmem_anchor/init_seed_tie`) — is the one with
/// WT = 70.
const TIES: &str = "\
ID,TIME,DV,MDV,EVID,AMT,CMT,WT
1,0,1,0,0,0,1,70
1,0,1,0,0,0,1,140
1,2,1,0,0,0,1,140
2,0,0,1,2,0,1,70
2,0,0,1,2,0,1,140
2,2,1,0,0,0,1,140
3,0,0,1,1,100,1,70
3,0,0,1,1,100,1,140
3,2,1,0,0,0,1,140
4,0,0,1,2,0,1,70
4,0,1,0,0,0,1,140
4,2,1,0,0,0,1,140
5,0,1,0,0,0,1,70
5,0,0,1,2,0,1,140
5,2,1,0,0,0,1,140
6,0,0,1,1,100,1,70
6,0,1,0,0,0,1,140
6,2,1,0,0,0,1,140
7,0,1,0,0,0,1,70
7,0,0,1,1,100,1,140
7,2,1,0,0,0,1,140
8,0,0,1,1,100,1,70
8,0,0,1,2,0,1,140
8,2,1,0,0,0,1,140
9,0,0,1,2,0,1,70
9,0,0,1,1,100,1,140
9,2,1,0,0,0,1,140
";

fn wt(s: &Subject, rec: FirstRecord) -> f64 {
    let cov = match rec {
        FirstRecord::Dose(k) => s.dose_cov(k),
        FirstRecord::PkOnly(m) => s.pk_only_cov(m),
        FirstRecord::Obs(j) => s.obs_cov(j),
    };
    cov["WT"]
}

fn kind(rec: FirstRecord) -> &'static str {
    match rec {
        FirstRecord::Dose(_) => "dose",
        FirstRecord::PkOnly(_) => "pk-only",
        FirstRecord::Obs(_) => "obs",
    }
}

/// #2121: the selector returns the first record in file order for every pair of kinds
/// and both file orders, on the reader's own output (so after its #614/#1810 ULP
/// moves, which are what turn file order into the walk's order).
#[test]
fn first_record_is_first_in_file_order_for_every_kind_pair() {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(TIES.as_bytes()).unwrap();
    let pop = crate::read_nonmem_csv(f.path(), None, None).unwrap();
    // (ID, kind of the first row in file order).
    let want = [
        ("1", "obs"),
        ("2", "pk-only"),
        ("3", "dose"),
        ("4", "pk-only"),
        ("5", "obs"),
        ("6", "dose"),
        ("7", "obs"),
        ("8", "dose"),
        ("9", "pk-only"),
    ];
    for (id, want_kind) in want {
        let s = pop.subjects.iter().find(|s| s.id == id).expect(id);
        let rec = first_record(s, s.doses.len()).expect("records");
        assert_eq!(
            (kind(rec), wt(s, rec)),
            (want_kind, 70.0),
            "ID {id}: the seed must be the tie's first row in file order"
        );
    }
    // The straddle: each mixed pair is in the file in both orders, and the two orders
    // pick records of different kinds. A rule that ignored file order (a fixed kind
    // preference, or last-wins) cannot satisfy both halves of a pair.
    for (a, b) in [("4", "5"), ("6", "7"), ("8", "9")] {
        let k = |id: &str| {
            let s = pop.subjects.iter().find(|s| s.id == id).unwrap();
            kind(first_record(s, s.doses.len()).unwrap())
        };
        assert_ne!(k(a), k(b), "IDs {a}/{b}: the two file orders pick one kind");
    }
}

/// `n_doses` caps the dose candidates (the adaptive replay's appended controller doses
/// are not records), and a subject with no record has no seed.
#[test]
fn first_record_skips_doses_past_n_doses_and_is_none_without_records() {
    let mut s = Subject { id: "1".into(), obs_times: vec![1.0], ..Default::default() };
    s.doses.push(crate::types::DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0));
    assert_eq!(first_record(&s, 1), Some(FirstRecord::Dose(0)));
    assert_eq!(first_record(&s, 0), Some(FirstRecord::Obs(0)));
    let empty = Subject::default();
    assert_eq!(first_record(&empty, 0), None);
}
