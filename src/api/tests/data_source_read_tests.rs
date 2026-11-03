//! #2140: a population read from a [`DataSource`] is the population a read of the
//! file's path gives, and the entry points stamp `data_hash` from the bytes they
//! parsed rather than from a second read of the path.

use crate::io::hash::sha256_file;
use crate::{DataSource, ModelSource, Population, ReaderSettings};

const MODEL: &str = "[parameters]\n  theta TVCL(1.0, 0.1, 10.0)\n  theta TVV(10.0, 1.0, 100.0)\n  \
     omega ETA_CL ~ 0.1\n  sigma PROP ~ 0.1\n\n[individual_parameters]\n  \
     CL = TVCL * exp(ETA_CL)\n  V = TVV\n\n[structural_model]\n  \
     pk one_cpt_iv(cl=CL, v=V)\n\n[error_model]\n  DV ~ proportional(PROP)\n";

const CSV: &str = "ID,TIME,DV,EVID,AMT,CMT\n\
                   1,0,.,1,100,1\n1,1,8.0,0,.,1\n1,4,5.0,0,.,1\n\
                   2,0,.,1,100,1\n2,1,9.0,0,.,1\n2,4,6.0,0,.,1\n\
                   3,0,.,1,100,1\n3,1,7.0,0,.,1\n3,4,4.0,0,.,1\n";

struct Files {
    _dir: tempfile::TempDir,
    model: std::path::PathBuf,
    data: std::path::PathBuf,
}

fn files(csv: &str) -> Files {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("m.ferx");
    let data = dir.path().join("d.csv");
    std::fs::write(&model, MODEL).unwrap();
    std::fs::write(&data, csv).unwrap();
    Files {
        _dir: dir,
        model,
        data,
    }
}

fn path(p: &std::path::Path) -> &str {
    p.to_str().unwrap()
}

/// The settings with a row filter, so a read that dropped them would differ.
fn settings(parsed: &crate::types::ParsedModel) -> ReaderSettings {
    let mut s = ReaderSettings::from_parsed(parsed);
    s.ignore_subjects = vec!["2".to_string()];
    s
}

fn ids(p: &Population) -> Vec<String> {
    p.subjects.iter().map(|s| s.id.clone()).collect()
}

/// A source read from the file, and one built from the same bytes in memory, give
/// the population the path read gives — with the settings applied: subject 2 is
/// filtered out on all three.
///
/// Mutation — read nothing (or the file's path) for `CsvInput::Bytes`, or drop the
/// settings on the source arm: the `Debug` equality and the `ids` assertion.
#[test]
fn a_source_reads_the_population_its_path_reads() {
    let f = files(CSV);
    let parsed = ModelSource::read(&f.model).unwrap().parsed;
    let s = settings(&parsed);
    let by_path = crate::read_population_with(&parsed.model, &s, path(&f.data)).unwrap();
    assert_eq!(ids(&by_path.0), ["1", "3"]);

    let from_file = DataSource::read(&f.data).unwrap();
    let from_mem = DataSource::from_bytes(CSV);
    for (what, src) in [("file", &from_file), ("memory", &from_mem)] {
        let got = crate::read_population_with_source(&parsed.model, &s, src).unwrap();
        assert_eq!(ids(&got.0), ["1", "3"], "{what}");
        assert_eq!(format!("{:?}", got.0), format!("{:?}", by_path.0), "{what}");
    }
}

/// The population is the source's bytes, not the file's current contents: a CSV
/// rewritten after the read does not reach it.
#[test]
fn an_edit_after_the_read_does_not_reach_the_population() {
    let f = files(CSV);
    let parsed = ModelSource::read(&f.model).unwrap().parsed;
    let s = ReaderSettings::from_parsed(&parsed);
    let src = DataSource::read(&f.data).unwrap();
    std::fs::write(
        &f.data,
        "ID,TIME,DV,EVID,AMT,CMT\n9,0,.,1,100,1\n9,1,8.0,0,.,1\n",
    )
    .unwrap();
    let got = crate::read_population_with_source(&parsed.model, &s, &src).unwrap();
    assert_eq!(ids(&got.0), ["1", "2", "3"]);
    let now = crate::read_population_with(&parsed.model, &s, path(&f.data)).unwrap();
    assert_eq!(ids(&now.0), ["9"]);
}

/// A dataset the reader refuses is refused the same way from a source: the same
/// code and the same message, byte for byte.
#[test]
fn a_refused_dataset_is_refused_alike_from_a_source() {
    let f = files("ID,DV,EVID,AMT\n1,.,1,100\n1,8.0,0,.\n");
    let parsed = ModelSource::read(&f.model).unwrap().parsed;
    let s = ReaderSettings::from_parsed(&parsed);
    let by_path = crate::read_population_with(&parsed.model, &s, path(&f.data))
        .err()
        .expect("no TIME column");
    let src = DataSource::read(&f.data).unwrap();
    let by_src = crate::read_population_with_source(&parsed.model, &s, &src)
        .err()
        .expect("no TIME column");
    assert_eq!(by_src.code(), Some("E_DATA"));
    assert_eq!(by_src.code(), by_path.code());
    assert_eq!(by_src.to_string(), by_path.to_string());
}

/// The hashed read the entry points stamp `data_hash` from: the population is the
/// path read's, and the hash is the file's. Its failures are the path read's, in
/// the same order — a clause that does not compile is reported before a missing
/// file, and a missing file with the reader's own message.
///
/// Mutation — read the file before compiling the filter: the `E_PARSE` assertion.
#[test]
fn the_hashed_read_is_the_path_read_and_the_files_hash() {
    let f = files(CSV);
    let parsed = ModelSource::read(&f.model).unwrap().parsed;
    let s = settings(&parsed);
    let (pop, _, hash) =
        super::run::read_population_hashed_diag(&parsed.model, &s, path(&f.data)).unwrap();
    let by_path = crate::read_population_with(&parsed.model, &s, path(&f.data)).unwrap();
    assert_eq!(format!("{pop:?}"), format!("{:?}", by_path.0));
    assert_eq!(hash, sha256_file(&f.data).unwrap());

    let missing = f.data.with_file_name("missing.csv");
    let want = super::run::read_population_with_diag(&parsed.model, &s, path(&missing))
        .err()
        .expect("missing");
    let got = super::run::read_population_hashed_diag(&parsed.model, &s, path(&missing))
        .err()
        .expect("missing");
    assert_eq!(got.code, "E_DATA");
    assert!(
        got.message.starts_with("Failed to open CSV: "),
        "{}",
        got.message
    );
    assert_eq!(got.message, want.message);

    let mut bad = s.clone();
    bad.ignore_exprs = vec!["DV >".to_string()];
    let got = super::run::read_population_hashed_diag(&parsed.model, &bad, path(&missing))
        .err()
        .expect("bad clause");
    let want = super::run::read_population_with_diag(&parsed.model, &bad, path(&missing))
        .err()
        .expect("bad clause");
    assert_eq!(got.code, "E_PARSE");
    assert_eq!(got.message, want.message);
}

/// `prepare_run` stamps `data_hash` from the read it parsed: the file's SHA-256,
/// and never `None` for a dataset that was read.
#[test]
fn prepare_run_stamps_the_hash_of_the_read() {
    let f = files(CSV);
    let run = crate::prepare_run(path(&f.model), Some(path(&f.data))).unwrap();
    assert_eq!(run.data_hash, Some(sha256_file(&f.data).unwrap()));
    assert_eq!(ids(&run.population), ["1", "2", "3"]);
}
