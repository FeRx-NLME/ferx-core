use super::*;

fn names() -> Vec<String> {
    vec!["CL".to_string(), "V".to_string()]
}

fn subject_ids() -> Vec<String> {
    vec!["A".to_string(), "B".to_string(), "C".to_string()]
}

fn replicate(index: usize) -> ReplicateResult {
    ReplicateResult {
        index,
        estimates: vec![1.0 + index as f64, 10.0 + index as f64],
        standard_errors: None,
        ofv: 100.0 + index as f64,
        converged: true,
        estimate_near_boundary: false,
        covariance_step_successful: false,
        covariance_step_warnings: false,
        seconds: 0.25,
        error: None,
        delta_ofv: None,
    }
}

fn draw(index: usize) -> Replicate {
    Replicate {
        index,
        keys: vec![0, index % 3, 2],
    }
}

fn read(path: &Path) -> Vec<Vec<String>> {
    csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_path(path)
        .expect("readable csv")
        .records()
        .map(|r| r.expect("record").iter().map(str::to_string).collect())
        .collect()
}

fn open(dir: &Path, kept: &[ReplicateResult], draws: &[Replicate]) -> Journal {
    Journal::create(
        dir,
        &names(),
        &subject_ids(),
        None,
        kept,
        draws,
        &BootstrapOptions::default(),
    )
    .expect("journal opens")
}

#[test]
fn a_fresh_journal_writes_only_the_two_headers() {
    let dir = tempfile::tempdir().expect("temp dir");
    let journal = open(dir.path(), &[], &[]);
    journal.into_result().expect("no write error");

    let raw = read(&dir.path().join("raw_results.csv"));
    assert_eq!(raw.len(), 1, "header only");
    assert_eq!(raw[0][0], "sample");
    assert_eq!(raw[0].last().map(String::as_str), Some("error"));

    // `sample_keys1.csv` names its columns; the other two draw files do not.
    assert_eq!(
        read(&dir.path().join("sample_keys1.csv")),
        vec![subject_ids()]
    );
    assert!(read(&dir.path().join("included_individuals1.csv")).is_empty());
    assert!(read(&dir.path().join("included_keys1.csv")).is_empty());
}

/// The row-order guarantee #1141 documents, now under an append-as-you-go
/// writer: replicate *j* must be at the same row of all four files.
#[test]
fn appended_rows_stay_aligned_across_the_four_files() {
    let dir = tempfile::tempdir().expect("temp dir");
    let journal = open(dir.path(), &[], &[]);
    // Deliberately out of index order: the journal records completion order,
    // and the alignment must not depend on the indices being sorted.
    for i in [3usize, 1, 2] {
        journal.append(&replicate(i), Some(&draw(i)));
    }
    journal.into_result().expect("no write error");

    let raw = read(&dir.path().join("raw_results.csv"));
    let individuals = read(&dir.path().join("included_individuals1.csv"));
    let keys = read(&dir.path().join("included_keys1.csv"));
    let counts = read(&dir.path().join("sample_keys1.csv"));

    assert_eq!(raw.len(), 4, "header + 3 replicates");
    assert_eq!(individuals.len(), 3);
    assert_eq!(keys.len(), 3);
    assert_eq!(counts.len(), 4, "header + 3 replicates");

    for (j, index) in [3usize, 1, 2].into_iter().enumerate() {
        assert_eq!(raw[j + 1][0], index.to_string(), "raw row {j}");
        let expected = draw(index);
        let ids: Vec<String> = expected
            .keys
            .iter()
            .map(|&k| subject_ids()[k].clone())
            .collect();
        assert_eq!(individuals[j], ids, "included_individuals row {j}");
        let ordinals: Vec<String> = expected.keys.iter().map(|&k| (k + 1).to_string()).collect();
        assert_eq!(keys[j], ordinals, "included_keys row {j}");
        let n: Vec<String> = expected
            .counts(subject_ids().len())
            .iter()
            .map(|c| c.to_string())
            .collect();
        assert_eq!(counts[j + 1], n, "sample_keys row {j}");
    }
}

/// The base fit has no draw, so it gets a `raw_results` row and nothing else —
/// otherwise the three draw files would gain a phantom row and every replicate
/// after it would be off by one.
#[test]
fn the_base_fit_writes_no_draw_rows() {
    let dir = tempfile::tempdir().expect("temp dir");
    let journal = open(dir.path(), &[], &[]);
    journal.append(&replicate(0), None);
    journal.append(&replicate(1), Some(&draw(1)));
    journal.into_result().expect("no write error");

    assert_eq!(read(&dir.path().join("raw_results.csv")).len(), 3);
    assert_eq!(
        read(&dir.path().join("included_individuals1.csv")).len(),
        1,
        "only the replicate has a draw"
    );
}

/// Opening the journal on a resume **rewrites** the kept rows rather than
/// appending to whatever bytes were there. That is what makes a truncated
/// trailing row self-healing: appending straight onto a record cut mid-field
/// would splice the fragment and the new row into one corrupt record.
#[test]
fn reopening_rewrites_the_kept_rows_over_a_damaged_file() {
    let dir = tempfile::tempdir().expect("temp dir");
    let raw_path = dir.path().join("raw_results.csv");
    let journal = open(dir.path(), &[], &[]);
    for i in [1usize, 2] {
        journal.append(&replicate(i), Some(&draw(i)));
    }
    journal.into_result().expect("no write error");

    // Chop the last row mid-field, as a hard kill would.
    let text = std::fs::read_to_string(&raw_path).expect("read");
    std::fs::write(&raw_path, &text[..text.len() - 6]).expect("truncate");

    let draws = vec![draw(1), draw(2), draw(3)];
    let journal = open(dir.path(), &[replicate(1)], &draws);
    journal.append(&replicate(2), Some(&draw(2)));
    journal.into_result().expect("no write error");

    let raw = read(&raw_path);
    assert_eq!(raw.len(), 3, "header + the kept row + the refitted one");
    assert_eq!(raw[1][0], "1");
    assert_eq!(raw[2][0], "2");
    // And the file is well-formed again: every row is full width.
    assert!(raw.iter().all(|r| r.len() == raw[0].len()), "{raw:?}");
    assert_eq!(read(&dir.path().join("included_keys1.csv")).len(), 2);
    // The rewrite went through temp siblings; none of them are left behind.
    for name in [
        "raw_results.csv",
        "included_individuals1.csv",
        "included_keys1.csv",
        "sample_keys1.csv",
    ] {
        let part = dir.path().join(format!("{name}.part"));
        assert!(!part.exists(), "{} left behind", part.display());
    }
}

/// The headers have to reach disk when the journal is opened, not when the
/// first replicate finishes. A run killed during the base fit — often the
/// longest single fit — would otherwise leave a `raw_results.csv` that exists
/// but is empty, which `--resume` rejects as a malformed header instead of
/// reporting that there is nothing to resume.
#[test]
fn the_headers_reach_disk_before_the_first_append() {
    let dir = tempfile::tempdir().expect("temp dir");
    // Read while the journal is still open: dropping it would flush, so a
    // test that closes it first cannot see the difference.
    let journal = open(dir.path(), &[], &[]);

    let raw = read(&dir.path().join("raw_results.csv"));
    assert_eq!(raw.len(), 1, "header on disk already: {raw:?}");
    assert_eq!(raw[0][0], "sample");
    assert_eq!(
        read(&dir.path().join("sample_keys1.csv")),
        vec![subject_ids()]
    );

    journal.into_result().expect("no write error");
}

/// The rewrite must never be the only copy of the recovery data. When it fails
/// part-way through, the interrupted run's artefacts are still on disk exactly
/// as they were — the swap into place happens only once every file is complete.
#[test]
fn a_failed_reopen_leaves_the_previous_files_intact() {
    let dir = tempfile::tempdir().expect("temp dir");
    let raw_path = dir.path().join("raw_results.csv");
    let journal = open(dir.path(), &[], &[]);
    for i in [1usize, 2] {
        journal.append(&replicate(i), Some(&draw(i)));
    }
    journal.into_result().expect("no write error");
    let before = std::fs::read_to_string(&raw_path).expect("read");

    // A directory in the way of the third file's temp sibling: the rewrite
    // fails there, with the first two already written.
    std::fs::create_dir(dir.path().join("included_keys1.csv.part")).expect("mkdir");

    let err = Journal::create(
        dir.path(),
        &names(),
        &subject_ids(),
        None,
        &[replicate(1), replicate(2)],
        &[draw(1), draw(2)],
        &BootstrapOptions::default(),
    )
    .map(|_| ())
    .expect_err("the rewrite fails");
    assert!(err.contains("included_keys1.csv.part"), "{err}");

    assert_eq!(std::fs::read_to_string(&raw_path).expect("read"), before);
    assert!(
        !dir.path().join("raw_results.csv.part").exists(),
        "the temp siblings are cleaned up"
    );
}

/// `--keep-covariance` and `--dofv` widen the row. The journal has to use the
/// same shape the final rewrite will, or the columns would stop matching the
/// names they were written under — the one corruption a reader cannot detect.
#[test]
fn the_column_shape_follows_the_options() {
    let dir = tempfile::tempdir().expect("temp dir");
    let options = BootstrapOptions {
        keep_covariance: true,
        dofv: true,
        ..BootstrapOptions::default()
    };
    let journal = Journal::create(
        dir.path(),
        &names(),
        &subject_ids(),
        None,
        &[],
        &[],
        &options,
    )
    .expect("journal opens");
    let mut r = replicate(1);
    r.standard_errors = Some(vec![Some(0.1), None]);
    r.delta_ofv = Some(2.5);
    journal.append(&r, Some(&draw(1)));
    journal.into_result().expect("no write error");

    let raw = read(&dir.path().join("raw_results.csv"));
    assert!(raw[0].contains(&"se_CL".to_string()), "{:?}", raw[0]);
    assert!(raw[0].contains(&"delta_ofv".to_string()), "{:?}", raw[0]);
    assert_eq!(raw[1].len(), raw[0].len());
    // The unreported SE is an empty cell, not a zero: it must come back as
    // `None` so the column stays aligned with its name.
    let se_v = raw[0]
        .iter()
        .position(|h| h == "se_V")
        .expect("se_V column");
    assert_eq!(raw[1][se_v], "");
}

/// A fit error is written on one line. The recovery rule — drop an unterminated
/// trailing record — depends on every row being exactly one line.
#[test]
fn a_multiline_error_is_flattened_to_one_row() {
    let dir = tempfile::tempdir().expect("temp dir");
    let journal = open(dir.path(), &[], &[]);
    let mut r = replicate(1);
    r.error = Some("line one\nline two\r\nline three".to_string());
    r.estimates = Vec::new();
    journal.append(&r, Some(&draw(1)));
    journal.into_result().expect("no write error");

    let text = std::fs::read_to_string(dir.path().join("raw_results.csv")).expect("read");
    assert_eq!(text.lines().count(), 2, "header + one row: {text}");
    assert!(text.contains("line one line two  line three"), "{text}");
}
