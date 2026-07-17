use super::grid_median_from_cumhaz;

#[test]
fn grid_median_interpolates_and_handles_no_crossing() {
    let ln2 = std::f64::consts::LN_2;
    let grid = [0.0, 1.0, 2.0, 3.0];
    // H crosses ln2 between t=1 and t=2; linear interp puts the median at t=1.5
    // (frac = (ln2 − 0.5·ln2)/(1.5·ln2 − 0.5·ln2) = 0.5).
    let cum = [0.0, ln2 * 0.5, ln2 * 1.5, ln2 * 3.0];
    let m = grid_median_from_cumhaz(&grid, &cum);
    assert!(
        (m - 1.5).abs() < 1e-9,
        "median should interpolate to 1.5; got {m}"
    );
    // H never reaches ln2 on the grid → NaN.
    let never = grid_median_from_cumhaz(&grid, &[0.0, 0.1, 0.2, 0.3]);
    assert!(never.is_nan(), "no crossing → NaN; got {never}");
    // Median at/before the first grid point (H(grid[0]) ≥ ln2): interpolated from the
    // origin (0,0). grid[0]=2, H=2·ln2 ⇒ median = 2·ln2 / (2·ln2) = 1.0.
    let early = grid_median_from_cumhaz(&[2.0, 4.0], &[ln2 * 2.0, ln2 * 4.0]);
    assert!(
        (early - 1.0).abs() < 1e-9,
        "median before the first grid point interpolates from the origin; got {early}"
    );
}
