//! #2867: the multi-start summary on hand-built start tables, one test per cell of the
//! line's input space. Each asserts the whole line, so deleting any clause of it fails the
//! test of the cell that prints that clause.

use super::{StartRecord, StartSpread, START_AGREE_OFV};

/// A start that ran.
fn ran(start: usize, ofv: f64, converged: bool) -> StartRecord {
    StartRecord {
        start,
        ofv: Some(ofv).filter(|v| v.is_finite()),
        penalized: false,
        objective: None,
        converged,
        status: Some(
            if converged {
                "converged"
            } else {
                "not_converged"
            }
            .to_string(),
        ),
        status_reasons: Vec::new(),
        n_iterations: 10,
        wall_time_secs: 0.5,
        theta: vec![1.0, 2.0],
        selected: false,
        error: None,
        error_code: None,
    }
}

/// A start that returned no result.
fn failed(start: usize) -> StartRecord {
    StartRecord {
        ofv: None,
        status: None,
        theta: Vec::new(),
        n_iterations: 0,
        wall_time_secs: 0.0,
        error: Some("refused".into()),
        error_code: Some("E_X".into()),
        ..ran(start, 0.0, false)
    }
}

/// Converged starts at the given OFVs, `selected` marked.
fn converged(ofvs: &[f64], selected: usize) -> Vec<StartRecord> {
    select(
        ofvs.iter()
            .enumerate()
            .map(|(k, v)| ran(k, *v, true))
            .collect(),
        selected,
    )
}

fn select(mut starts: Vec<StartRecord>, selected: usize) -> Vec<StartRecord> {
    starts[selected].selected = true;
    starts
}

fn line(starts: &[StartRecord]) -> String {
    StartSpread::of(starts).line()
}

/// Run A of #2867: eight starts on one optimum. Start 6 is lower by 1e-7, which is not a
/// better optimum, and the line has to say so.
#[test]
fn all_starts_agreeing_is_said_in_so_many_words() {
    let starts = converged(
        &[
            -221.877047838,
            -221.877047839,
            -221.877047829,
            -221.877047836,
            -221.877047894,
            -221.877047901,
            -221.877047924,
            -221.877047837,
        ],
        6,
    );
    let s = StartSpread::of(&starts);
    assert_eq!(
        (s.n_starts, s.n_converged, s.n_compared, s.n_within),
        (8, 8, 8, 8)
    );
    assert_eq!((s.n_not_converged, s.n_failed, s.n_unusable), (0, 0, 0));
    assert!(s.above.is_empty() && s.lower_not_converged.is_empty());
    let range = s.range.expect("eight compared starts have a range");
    assert!(
        range.is_finite() && (range - 9.5e-8).abs() < 1e-12,
        "{range}"
    );
    assert_eq!(
        s.line(),
        "Multi-start: 8 starts, 8 converged; best OFV -221.8770 (start 6); \
         all 8 converged starts within 1.0 of best (range 0.0000)"
    );
}

/// The line is the same shape when the model file's own start wins: only the index moves.
#[test]
fn start_zero_winning_names_start_zero() {
    let starts = converged(&[10.0, 10.0, 10.25], 0);
    assert_eq!(
        line(&starts),
        "Multi-start: 3 starts, 3 converged; best OFV 10.0000 (start 0); \
         all 3 converged starts within 1.0 of best (range 0.2500)"
    );
}

/// Run B of #2867: start 4 reports converged 5.27 above the other seven.
#[test]
fn one_converged_stray_is_named_with_its_distance() {
    let mut ofvs = [-221.877048; 8];
    ofvs[2] = -221.877049;
    ofvs[4] = -216.603223;
    let starts = converged(&ofvs, 2);
    let s = StartSpread::of(&starts);
    assert_eq!((s.n_compared, s.n_within), (8, 7));
    assert_eq!(s.above.len(), 1);
    assert_eq!(s.above[0].0, 4);
    let range = s.range.expect("range");
    assert!(
        range.is_finite() && (range - 5.273826).abs() < 1e-9,
        "{range}"
    );
    assert!((s.above[0].1 - range).abs() < 1e-12);
    assert_eq!(
        s.line(),
        "Multi-start: 8 starts, 8 converged; best OFV -221.8770 (start 2); \
         7 of 8 converged starts within 1.0 of best; 1 converged above it: start 4 (+5.27)"
    );
}

/// Both sides of the agreement gate: a start exactly `START_AGREE_OFV` above the best
/// agrees, one just past it does not.
#[test]
fn the_agreement_cutoff_is_inclusive_at_one_ofv() {
    assert_eq!(START_AGREE_OFV, 1.0);
    let at = StartSpread::of(&converged(&[0.0, 1.0], 0));
    assert_eq!((at.n_within, at.above.len()), (2, 0));
    assert_eq!(
        at.line(),
        "Multi-start: 2 starts, 2 converged; best OFV 0.0000 (start 0); \
         all 2 converged starts within 1.0 of best (range 1.0000)"
    );
    let past = StartSpread::of(&converged(&[0.0, 1.0 + 1e-9], 0));
    assert_eq!((past.n_within, past.above.len()), (1, 1));
    assert_eq!(
        past.line(),
        "Multi-start: 2 starts, 2 converged; best OFV 0.0000 (start 0); \
         1 of 2 converged starts within 1.0 of best; 1 converged above it: start 1 (+1.00)"
    );
}

/// Not-converged and failed starts are counted in the head and kept out of the comparison:
/// the not-converged start 12 OFV above the best does not make the converged ones disagree.
#[test]
fn not_converged_and_failed_starts_are_counted_and_not_compared() {
    let starts = select(
        vec![
            ran(0, 50.0, true),
            ran(1, 62.0, false),
            failed(2),
            ran(3, 50.5, true),
            ran(4, 51.0, false),
        ],
        0,
    );
    let s = StartSpread::of(&starts);
    assert_eq!((s.n_converged, s.n_not_converged, s.n_failed), (2, 2, 1));
    assert_eq!((s.n_compared, s.n_within), (2, 2));
    assert_eq!(
        s.line(),
        "Multi-start: 5 starts, 2 converged, 2 not converged, 1 failed; \
         best OFV 50.0000 (start 0); all 2 converged starts within 1.0 of best (range 0.5000)"
    );
    // The other side of each head gate: with neither, neither is mentioned.
    assert_eq!(
        line(&converged(&[50.0, 50.5], 0)),
        "Multi-start: 2 starts, 2 converged; best OFV 50.0000 (start 0); \
         all 2 converged starts within 1.0 of best (range 0.5000)"
    );
}

/// A start with a `NaN` OFV (recorded as `None`) or the ~1e20 divergence sentinel is not an
/// objective: it is counted on its own and left out of the count and the range, which the
/// sentinel would otherwise set to 1e20.
#[test]
fn a_nan_or_sentinel_ofv_is_counted_apart_and_kept_out_of_the_range() {
    let starts = select(
        vec![
            ran(0, 5.0, true),
            ran(1, f64::NAN, true),
            ran(2, 1e20, true),
            ran(3, 5.5, true),
        ],
        0,
    );
    assert_eq!(
        starts[1].ofv, None,
        "premise: a NaN OFV is recorded as None"
    );
    let s = StartSpread::of(&starts);
    assert_eq!((s.n_converged, s.n_unusable, s.n_compared), (4, 2, 2));
    let range = s.range.expect("range");
    assert!(range.is_finite() && range == 0.5, "{range}");
    assert_eq!(
        s.line(),
        "Multi-start: 4 starts, 4 converged; best OFV 5.0000 (start 0); \
         all 2 converged starts within 1.0 of best (range 0.5000); 2 without a usable OFV"
    );
}

/// No start converged: the returned start is marked, and the comparison is over every start
/// with a usable value, since there are no converged ones to compare.
#[test]
fn no_converged_start_compares_all_starts_and_marks_the_returned_one() {
    let starts = select(
        vec![
            ran(0, 10.5, false),
            ran(1, 10.25, false),
            ran(2, 10.0, false),
            ran(3, 17.5, false),
        ],
        2,
    );
    let s = StartSpread::of(&starts);
    assert_eq!((s.n_converged, s.n_compared, s.n_within), (0, 4, 3));
    assert_eq!(
        s.line(),
        "Multi-start: 4 starts, 0 converged, 4 not converged; \
         best OFV 10.0000 (start 2, not converged); 3 of 4 starts within 1.0 of best; \
         1 above it: start 3 (+7.50)"
    );
}

/// Selection prefers a converged start, so a not-converged one can stop lower than the
/// start that was returned. Both sides of the gate: below is named, above is not.
#[test]
fn a_not_converged_start_below_the_returned_one_is_named() {
    let below = select(
        vec![
            ran(0, 100.0, true),
            ran(1, 100.0, true),
            ran(2, 96.79, false),
        ],
        0,
    );
    let s = StartSpread::of(&below);
    assert_eq!(s.lower_not_converged.len(), 1);
    assert_eq!(s.lower_not_converged[0].0, 2);
    assert_eq!(
        s.line(),
        "Multi-start: 3 starts, 2 converged, 1 not converged; best OFV 100.0000 (start 0); \
         all 2 converged starts within 1.0 of best (range 0.0000); \
         1 not converged at a lower OFV: start 2 (-3.21)"
    );
    let above = select(
        vec![
            ran(0, 100.0, true),
            ran(1, 100.0, true),
            ran(2, 103.21, false),
        ],
        0,
    );
    assert_eq!(
        line(&above),
        "Multi-start: 3 starts, 2 converged, 1 not converged; best OFV 100.0000 (start 0); \
         all 2 converged starts within 1.0 of best (range 0.0000)"
    );
}

/// A VI fit under the default `vi_final_ofv = none` has no OFV on any start. The line says
/// there is nothing to compare instead of printing a count over nothing.
#[test]
fn no_usable_ofv_on_the_returned_start_says_there_is_nothing_to_compare() {
    let starts = select((0..4).map(|k| ran(k, f64::NAN, true)).collect(), 0);
    let s = StartSpread::of(&starts);
    assert_eq!((s.best, s.n_compared, s.range), (None, 0, None));
    assert_eq!(
        s.line(),
        "Multi-start: 4 starts, 4 converged; no OFV to compare (start 0 returned)"
    );
}

/// Under `nn_l2` / `nn_smooth` the starts are ranked on the penalized objective. The numbers
/// are #2867's `nn_l2 = 3` run: start 3 is 0.5 above the winner in reported OFV and 25.24
/// above it in the objective that ranked it. The line follows the ranking, and says which
/// value it is printing; a summary taken on `ofv` would read "all 4 … within 1.0".
#[test]
fn a_penalized_fit_is_compared_on_the_objective_that_ranked_it() {
    let mut starts = converged(&[-1292.5264, -1292.5264, -1292.5264, -1292.0264], 1);
    for (s, objective) in starts
        .iter_mut()
        .zip([-1265.3763, -1265.3763, -1265.3763, -1240.1363])
    {
        s.penalized = true;
        s.objective = Some(objective);
    }
    let s = StartSpread::of(&starts);
    assert!(s.penalized);
    assert_eq!((s.n_compared, s.n_within), (4, 3));
    assert_eq!(
        s.line(),
        "Multi-start: 4 starts, 4 converged; best penalized objective -1265.3763 (start 1); \
         3 of 4 converged starts within 1.0 of best; 1 converged above it: start 3 (+25.24)"
    );
    // The same table, not penalized, reads the reported OFV.
    for s in &mut starts {
        s.penalized = false;
        s.objective = None;
    }
    assert_eq!(
        line(&starts),
        "Multi-start: 4 starts, 4 converged; best OFV -1292.5264 (start 1); \
         all 4 converged starts within 1.0 of best (range 0.5000)"
    );
}

/// A penalized table in which no start has a finite penalized objective. The ranking had
/// nothing to compare, so it returned the first converged start, and the summary must not
/// fall back to the OFVs it was never ranked on: here that would print "best OFV 10.0000
/// (start 0)" over a converged start 5 OFV lower. Both sides of the gate: the same rows not
/// penalized are compared on their OFV.
#[test]
fn a_penalized_table_with_no_finite_objective_is_not_compared_on_ofv() {
    let mut starts = converged(&[10.0, 5.0, 10.5], 0);
    for s in &mut starts {
        s.penalized = true;
    }
    let s = StartSpread::of(&starts);
    assert!(s.penalized);
    assert_eq!((s.best, s.n_compared, s.n_unusable), (None, 0, 3));
    assert_eq!(
        s.line(),
        "Multi-start: 3 starts, 3 converged; no penalized objective to compare (start 0 returned)"
    );
    for s in &mut starts {
        s.penalized = false;
    }
    let s = StartSpread::of(&starts);
    assert!(!s.penalized);
    assert_eq!(s.best, Some(10.0));
    assert!(
        s.line().contains("best OFV 10.0000 (start 0)"),
        "{}",
        s.line()
    );
}

/// More strays than the line names: the first three by start index, then a count. Both
/// sides: exactly three are all named, with no "more".
#[test]
fn strays_past_the_third_are_counted_not_listed() {
    let five = converged(&[0.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0);
    assert_eq!(
        line(&five),
        "Multi-start: 6 starts, 6 converged; best OFV 0.0000 (start 0); \
         1 of 6 converged starts within 1.0 of best; 5 converged above it: \
         start 1 (+2.00), start 2 (+3.00), start 3 (+4.00), and 2 more"
    );
    let three = converged(&[0.0, 2.0, 3.0, 4.0], 0);
    assert_eq!(
        line(&three),
        "Multi-start: 4 starts, 4 converged; best OFV 0.0000 (start 0); \
         1 of 4 converged starts within 1.0 of best; 3 converged above it: \
         start 1 (+2.00), start 2 (+3.00), start 3 (+4.00)"
    );
}

/// One converged start among failures has nothing to agree with; "all 1 … within 1.0" would
/// read as agreement.
#[test]
fn a_single_converged_start_has_nothing_to_compare_with() {
    let starts = select(vec![ran(0, 7.0, true), failed(1), ran(2, 9.0, false)], 0);
    assert_eq!(
        line(&starts),
        "Multi-start: 3 starts, 1 converged, 1 not converged, 1 failed; \
         best OFV 7.0000 (start 0); no other converged start to compare it with"
    );
}

/// A table with no selected row (a hand-edited bundle) still summarises its counts.
#[test]
fn a_table_with_no_selected_row_reports_its_counts_only() {
    let mut starts = converged(&[1.0, 2.0], 0);
    starts[0].selected = false;
    assert_eq!(line(&starts), "Multi-start: 2 starts, 2 converged");
}
