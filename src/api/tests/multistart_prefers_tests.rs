use super::multistart_prefers;

const SENTINEL: f64 = 2.3e16; // block_sigma-indefinite divergence OFV

#[test]
fn valid_unconverged_beats_invalid_converged() {
    // The regression: a diverged start reporting `converged = true` with a
    // sentinel OFV must NOT beat the valid, unconverged exact-inits start.
    assert!(multistart_prefers(SENTINEL, true, 867.0, false));
    assert!(!multistart_prefers(867.0, false, SENTINEL, true));
}

#[test]
fn non_finite_is_invalid() {
    assert!(multistart_prefers(f64::INFINITY, true, 900.0, false));
    assert!(multistart_prefers(f64::NAN, true, 900.0, false));
}

#[test]
fn among_valid_prefers_converged_then_lower_ofv() {
    // converged beats unconverged even at a (slightly) higher OFV.
    assert!(multistart_prefers(730.0, false, 735.0, true));
    // same convergence: lower OFV wins.
    assert!(multistart_prefers(740.0, true, 734.0, true));
    assert!(!multistart_prefers(734.0, true, 740.0, true));
}

#[test]
fn both_invalid_prefers_lower() {
    assert!(multistart_prefers(3e16, false, 2e16, false));
    assert!(!multistart_prefers(2e16, false, 3e16, false));
}
