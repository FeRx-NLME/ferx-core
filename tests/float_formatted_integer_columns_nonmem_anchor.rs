//! #1496 — a float-formatted whole number (`1.0`) in an integer data column reads
//! as that number, end to end through `fit()`, anchored on NONMEM reading the same
//! two files.
//!
//! # The object
//!
//! `tests/nonmem/warfarin_bloq_intcols.csv` and `warfarin_bloq_floatcols.csv` are
//! the warfarin M3 BLOQ dataset of `examples/warfarin_bloq.ferx` with `II`/`ADDL`
//! appended, identical but for how they spell four integer columns — `EVID`, `MDV`,
//! `CENS`, `ADDL` — as `1` / `0` / `2` or as `1.0` / `0.0` / `2.0`. pandas writes
//! the second form for a whole integer column once any cell in it is blank, and
//! ferx's own sdtab writes `CENS` that way. Before #1496 the reader took each of
//! those cells as 0: every dose became an unscored observation, the `ADDL` train
//! collapsed to its first dose, the `MDV=1` observation was scored, and the
//! censored rows were scored as measurements at the LLOQ.
//!
//! # Why the fixture can see it
//!
//! Each column is live in the integer file, and the test asserts that rather than
//! assuming it. The first `MDV` probe for this issue float-formatted a column whose
//! every `MDV=1` row was a dose row — where the flag decides nothing — and no number
//! moved. So:
//!
//! - `ADDL=2, II=24` on every dose row: removing the column moves the objective;
//! - 10 observation rows carry `CENS=1`: zeroing the column moves it;
//! - the observation at `ID 1, TIME 0.5` carries `MDV=1` with its real `DV`:
//!   clearing that one flag moves it;
//! - the `EVID=1` rows are the only doses there are.
//!
//! # NONMEM
//!
//! Both files were run through the committed control streams on NONMEM 7.6.0
//! (`tests/nonmem/warfarin_bloq_{intcols,floatcols}.{ctl,lst}`, each naming its own
//! dataset): 109 observation records and `#OBJV 279.35595553971399` on both, with
//! identical final estimates; with that one `MDV` flag cleared, 110 records and
//! 281.06112134711015. NONMEM reads the float spelling as the integer, and so must
//! ferx. The two engines' objectives are not compared with each other — this is
//! FOCEI against NONMEM's LAPLACE `F_FLAG` likelihood, which carries a different
//! additive constant (see `tests/bloq_convergence.rs`) — the anchor is that each
//! engine reads both files identically. `the_committed_nonmem_runs_read_both_files_alike`
//! checks the committed `.lst` pair says so, so the evidence cannot quietly become
//! two runs of one file (the #1009 review found exactly that hole in an earlier
//! anchor).
//!
//! Fast: `outer_maxiter = 0` on 10 subjects, so it runs on every PR. The engine is
//! beside the point: both files must produce the same `Population`, which is
//! asserted directly before any objective is compared.

use ferx_core::{fit, parse_full_model_file, read_nonmem_csv, Population};
use std::path::Path;

fn repo(rel: &str) -> String {
    format!("{}/{rel}", env!("CARGO_MANIFEST_DIR"))
}

/// The integer-spelled anchor dataset, line by line.
fn intcols_lines() -> Vec<String> {
    std::fs::read_to_string(repo("tests/nonmem/warfarin_bloq_intcols.csv"))
        .expect("the integer anchor dataset is committed")
        .lines()
        .map(str::to_string)
        .collect()
}

/// The four columns the float file spells as `1.0` / `0.0` / `2.0`.
const FLOAT_SPELLED: [&str; 4] = ["EVID", "MDV", "CENS", "ADDL"];

/// The float file is the integer file with exactly those four columns float-spelled.
///
/// It is an external file, so without this the twin can quietly become integer
/// against integer while every assertion below stays green and their messages go
/// on naming all four columns. Review round 1 of #1502 measured exactly that: the
/// float file with its 120 `MDV` cells rewritten as integers left both tests green,
/// and so does the same rewrite of any one of the four columns.
fn assert_float_file_is_the_integer_file_float_spelled() {
    let int_lines = intcols_lines();
    let float_lines: Vec<String> =
        std::fs::read_to_string(repo("tests/nonmem/warfarin_bloq_floatcols.csv"))
            .expect("the float anchor dataset is committed")
            .lines()
            .map(str::to_string)
            .collect();
    assert_eq!(
        float_lines[0], int_lines[0],
        "the two files share one header"
    );
    assert_eq!(float_lines.len(), int_lines.len(), "and the same rows");
    let header: Vec<&str> = int_lines[0].split(',').collect();
    for name in FLOAT_SPELLED {
        assert!(header.contains(&name), "the header carries {name}");
    }
    let rows = float_lines[1..].iter().zip(&int_lines[1..]);
    for (row, (float_line, int_line)) in rows.enumerate() {
        let f: Vec<&str> = float_line.split(',').collect();
        let i: Vec<&str> = int_line.split(',').collect();
        assert_eq!(f.len(), header.len(), "float file, data row {row}");
        assert_eq!(i.len(), header.len(), "integer file, data row {row}");
        for (k, &name) in header.iter().enumerate() {
            if FLOAT_SPELLED.contains(&name) {
                assert!(
                    i[k].parse::<i64>().is_ok(),
                    "{name}, data row {row}: the integer file spells `{}`, not an integer",
                    i[k]
                );
                assert!(
                    f[k].parse::<i64>().is_err(),
                    "{name}, data row {row}: the float file spells `{}` as an integer",
                    f[k]
                );
                assert_eq!(
                    f[k].parse::<f64>().ok(),
                    i[k].parse::<f64>().ok(),
                    "{name}, data row {row}: `{}` and `{}` must be the same number",
                    f[k],
                    i[k]
                );
            } else {
                assert_eq!(
                    f[k], i[k],
                    "{name}, data row {row}: every other column is byte-identical"
                );
            }
        }
    }
}

/// Rewrite one column of every data row of the integer file.
fn with_column(col: usize, cell: impl Fn(&[&str]) -> String) -> String {
    let lines = intcols_lines();
    let mut out = vec![lines[0].clone()];
    for line in &lines[1..] {
        let mut fields: Vec<String> = line.split(',').map(str::to_string).collect();
        let view: Vec<&str> = line.split(',').collect();
        fields[col] = cell(&view);
        out.push(fields.join(","));
    }
    out.join("\n") + "\n"
}

fn read_text(csv: &str) -> Population {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("data.csv");
    std::fs::write(&path, csv).expect("write the derived dataset");
    read_nonmem_csv(&path, None, None).unwrap_or_else(|e| panic!("derived dataset loads: {e}"))
}

fn read_committed(name: &str) -> Population {
    read_nonmem_csv(
        Path::new(&repo(&format!("tests/nonmem/{name}"))),
        None,
        None,
    )
    .unwrap_or_else(|e| panic!("{name} loads: {e}"))
}

/// FOCEI objective of `examples/warfarin_bloq.ferx` at its initial estimates.
fn ofv(population: &Population, what: &str) -> f64 {
    let parsed = parse_full_model_file(Path::new(&repo("examples/warfarin_bloq.ferx")))
        .expect("the example model parses");
    let mut opts = parsed.fit_options.clone();
    opts.outer_maxiter = 0;
    opts.run_covariance_step = false;
    let result = fit(
        &parsed.model,
        population,
        &parsed.model.default_params,
        &opts,
    )
    .unwrap_or_else(|e| panic!("{what}: fit returns Ok: {e}"));
    assert!(
        result.ofv.is_finite(),
        "{what}: the objective must be finite before it is compared, got {}",
        result.ofv
    );
    result.ofv
}

/// Everything the likelihood reads from a subject, flattened for comparison.
fn fingerprint(population: &Population) -> Vec<String> {
    population
        .subjects
        .iter()
        .map(|s| {
            format!(
                "{} doses={:?} obs_times={:?} dv={:?} cens={:?}",
                s.id,
                s.doses
                    .iter()
                    .map(|d| (d.time, d.amt, d.cmt_1based()))
                    .collect::<Vec<_>>(),
                s.obs_times,
                s.observations,
                s.cens,
            )
        })
        .collect()
}

/// A9. The two spellings produce the same `Population` and the same objective to
/// the last bit, and each of the four columns is live — so the equality is a
/// statement about all four, not about the three that happened to matter.
#[test]
fn float_formatted_integer_columns_fit_like_the_integer_spelling() {
    assert_float_file_is_the_integer_file_float_spelled();
    let integer = read_committed("warfarin_bloq_intcols.csv");
    let float = read_committed("warfarin_bloq_floatcols.csv");

    let n_doses = |p: &Population| p.subjects.iter().map(|s| s.doses.len()).sum::<usize>();
    let n_cens = |p: &Population| {
        p.subjects
            .iter()
            .map(|s| s.cens.iter().filter(|&&c| c != 0).count())
            .sum::<usize>()
    };
    // The integer leg, as NONMEM reads it: 10 dose rows × (1 + ADDL 2), 109 scored
    // observations (110 rows less the MDV=1 one), 10 of them censored.
    assert_eq!(integer.subjects.len(), 10);
    assert_eq!(
        n_doses(&integer),
        30,
        "ADDL=2 expands each dose row to 3 doses"
    );
    assert_eq!(integer.n_obs(), 109, "the MDV=1 observation is excluded");
    assert_eq!(n_cens(&integer), 10);

    assert_eq!(
        fingerprint(&float),
        fingerprint(&integer),
        "`1.0` / `0.0` / `2.0` must read as `1` / `0` / `2` in EVID, MDV, CENS and ADDL"
    );
    let ofv_int = ofv(&integer, "integer");
    let ofv_float = ofv(&float, "float");
    assert_eq!(
        ofv_float.to_bits(),
        ofv_int.to_bits(),
        "same dataset spelled two ways: {ofv_float:.12} against {ofv_int:.12}"
    );

    // Liveness, one column at a time: each derived file differs from the integer
    // file in exactly one column, and each must move the objective.
    let no_addl = {
        let lines = intcols_lines();
        let body: Vec<String> = lines
            .iter()
            .map(|l| {
                l.rsplit_once(',')
                    .expect("ADDL is the last column")
                    .0
                    .to_string()
            })
            .collect();
        assert!(body[0].ends_with(",II"), "header without ADDL: {}", body[0]);
        body.join("\n") + "\n"
    };
    let zero_cens = with_column(8, |_| "0".to_string());
    let mdv_cleared = with_column(7, |f| {
        if f[0] == "1" && f[1] == "0.5" {
            assert_eq!(f[7], "1", "the fixture's MDV=1 observation");
            "0".to_string()
        } else {
            f[7].to_string()
        }
    });
    for (what, csv) in [
        ("ADDL removed", no_addl),
        ("CENS zeroed", zero_cens),
        ("MDV flag cleared", mdv_cleared),
    ] {
        let other = ofv(&read_text(&csv), what);
        println!("{what}: {other:.10} against the integer file's {ofv_int:.10}");
        assert_ne!(
            other.to_bits(),
            ofv_int.to_bits(),
            "{what} must change the objective, else that column is dead in this fixture"
        );
    }
}

/// The NONMEM half of the anchor, as committed. Each `.lst` must name its own
/// dataset (so it records which file it read), and the two must report the same
/// observation count and objective.
#[test]
fn the_committed_nonmem_runs_read_both_files_alike() {
    let read = |tag: &str| {
        let lst = std::fs::read_to_string(repo(&format!("tests/nonmem/warfarin_bloq_{tag}.lst")))
            .expect("the NONMEM listing is committed");
        assert!(
            lst.contains(&format!("$DATA warfarin_bloq_{tag}.csv")),
            "{tag}.lst must record that it read warfarin_bloq_{tag}.csv"
        );
        let field = |key: &str| -> String {
            lst.lines()
                .find(|l| l.contains(key))
                .unwrap_or_else(|| panic!("{tag}.lst has no `{key}` line"))
                .split(':')
                .next_back()
                .unwrap()
                .trim()
                .to_string()
        };
        (
            field("TOT. NO. OF OBS RECS"),
            field("OBJECTIVE FUNCTION VALUE WITHOUT CONSTANT"),
        )
    };
    let integer = read("intcols");
    let float = read("floatcols");
    // The dataset holds 110 observation records, so NONMEM's 109 is exactly the one
    // `MDV=1` observation excluded — the straddle stated in the test, not only in its
    // message.
    let lines = intcols_lines();
    let evid = lines[0].split(',').position(|h| h == "EVID").expect("EVID");
    let n_evid0 = lines[1..]
        .iter()
        .filter(|l| l.split(',').nth(evid) == Some("0"))
        .count();
    assert_eq!(n_evid0, 110, "observation records in the committed dataset");
    assert_eq!(
        integer.0,
        (n_evid0 - 1).to_string(),
        "NONMEM excludes exactly the MDV=1 observation"
    );
    assert_eq!(integer.1, "279.35595553971399");
    assert_eq!(
        float, integer,
        "NONMEM reads the float spelling as the integer"
    );
}
