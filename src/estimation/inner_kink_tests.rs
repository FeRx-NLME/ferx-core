//! Tier 1 for the non-smooth individual objective (#3159), on a V-shaped valley with a closed
//! form, where the end-to-end fixture (`tests/inner_kink_nonsmooth.rs`) has none.

use super::kink::{kick_restarts, kink_escape_allowed, min_norm_hull, stall_escape, STALL_GNORM};

/// `f = 10|x| + ½(y − 3)²`: a kink along `x = 0` whose floor falls toward `y = 3`. The gradient
/// on the floor is `(±10, y − 3)`, across the valley, and every gradient step crosses the kink.
fn valley(p: &[f64]) -> f64 {
    10.0 * p[0].abs() + 0.5 * (p[1] - 3.0).powi(2)
}

fn valley_grad(p: &[f64]) -> Vec<f64> {
    vec![10.0 * if p[0] >= 0.0 { 1.0 } else { -1.0 }, p[1] - 3.0]
}

/// The segment between `(3, 0)` and `(−1, 2)` has its minimum-norm point at `t = 0.6`, `(0.6, 1.2)`;
/// two opposite gradients contain zero; one gradient is its own hull.
#[test]
fn the_min_norm_point_of_a_hull_is_found() {
    let v = min_norm_hull(&[vec![3.0, 0.0], vec![-1.0, 2.0]]);
    assert!((v[0] - 0.6).abs() < 1e-9 && (v[1] - 1.2).abs() < 1e-9, "{v:?}");
    let z = min_norm_hull(&[vec![10.0, 2.0], vec![-10.0, 2.0], vec![0.0, -4.0]]);
    assert!(z.iter().all(|c| c.abs() < 1e-9), "{z:?}");
    assert_eq!(min_norm_hull(&[vec![1.5, -2.0]]), vec![1.5, -2.0]);
}

/// On the valley floor off its minimum the sampled gradients straddle the kink, the hull's
/// minimum-norm point drops the cross-valley component, and the escape moves along the valley.
/// At the minimum zero is in the hull (a kink minimum, certified); below the gradient threshold
/// nothing is sampled, whether the pre-step gradient or the stalled point's own is below it; a
/// non-finite gradient declines; and a non-finite objective ends the walk (#3159 review).
#[test]
fn a_stall_on_a_valley_floor_escapes_along_it_and_a_kink_minimum_is_certified() {
    let x = [1e-9, 5.0];
    let g = valley_grad(&x);
    let e = stall_escape(&valley, &valley_grad, &x, valley(&x), &g).expect("a descent exists");
    assert!(e.f < valley(&x) - 1e-3 && e.x[1] < 5.0 - 1e-3, "{:?} {}", e.x, e.f);
    assert_eq!(e.g, valley_grad(&e.x));

    let min = [1e-9, 3.0];
    assert!(stall_escape(&valley, &valley_grad, &min, valley(&min), &g).is_none());
    let small = [STALL_GNORM * 0.5, 0.0];
    assert!(stall_escape(&valley, &valley_grad, &x, valley(&x), &small).is_none());
    let flat = |p: &[f64]| valley(p) * 0.01;
    let flat_grad = |p: &[f64]| valley_grad(p).iter().map(|v| v * 0.01).collect();
    assert!(stall_escape(&flat, &flat_grad, &x, flat(&x), &g).is_none(), "stale pre-step norm");
    let nan = |_: &[f64]| vec![f64::NAN, 0.0];
    assert!(stall_escape(&valley, &nan, &x, valley(&x), &g).is_none());
    // NaN on a band just below `x`, lower beyond it: the walk stops at the band.
    let blows_up =
        |p: &[f64]| if (4.999..5.0 - 1e-6).contains(&p[1]) { f64::NAN } else { valley(p) };
    assert!(stall_escape(&blows_up, &valley_grad, &x, valley(&x), &g).is_none());
}

/// An identity `probe` (no descent) and `finish` (so the walk's own kicked point is what is
/// finally compared and adopted). A kick moves one informative coordinate by a quarter of its
/// prior SD and keeps a result lower by more than 1e-9; a placeholder coordinate (SD below
/// 1e-3) is not kicked, and each accepted result is the next kick's incumbent.
#[test]
fn a_kick_restart_keeps_only_a_lower_mode_and_skips_placeholder_coordinates() {
    let obj = |p: &[f64]| p.iter().map(|v| (v - 1.0).powi(2)).sum::<f64>();
    let x = [0.0, 0.0, 0.0];
    let mut kicked = Vec::new();
    let probe = |cand: &mut Vec<f64>| kicked.push(cand.clone());
    let r = kick_restarts(&x, obj(&x), &[2.0, 0.0, 4.0], &obj, probe, |_: &mut Vec<f64>| ());
    // The widest prior first: coordinate 2 kicks to +1.0 (lower, taken) and −1.0 (higher,
    // dropped); coordinate 1 is a placeholder; coordinate 0 kicks from the new incumbent to +0.5.
    let (best, f, ()) = r.expect("a kick finds a lower point");
    assert_eq!(best, vec![0.5, 0.0, 1.0]);
    assert!((f - obj(&best)).abs() < 1e-15);
    assert_eq!(kicked.len(), 4, "{kicked:?}");
    assert!(kicked.iter().all(|k| k[1] == 0.0), "the placeholder coordinate was kicked");

    let at_min = [1.0, 1.0, 1.0];
    let none = kick_restarts(
        &at_min,
        0.0,
        &[2.0, 2.0, 2.0],
        &obj,
        |_: &mut Vec<f64>| (),
        |_: &mut Vec<f64>| (),
    );
    assert!(none.is_none());
}

/// `finish` re-checks the walk's end point against the incumbent's objective, not `probe`'s: a
/// `finish` that moves the walk's end point back to `x` finds nothing to adopt, even though a
/// raw kicked point was lower (#3159 review).
#[test]
fn finish_decides_the_final_accept_reject_not_probe() {
    let obj = |p: &[f64]| p.iter().map(|v| (v - 1.0).powi(2)).sum::<f64>();
    let x = [0.0, 0.0];
    let mut probed_lower = false;
    let probe = |cand: &mut Vec<f64>| probed_lower |= obj(cand) + 1e-9 < obj(&x.to_vec());
    let finish = |cand: &mut Vec<f64>| cand.copy_from_slice(&[0.0, 0.0]); // undo the walk
    assert!(kick_restarts(&x, obj(&x), &[2.0, 2.0], &obj, probe, finish).is_none());
    assert!(probed_lower, "premise: a raw kick was lower");
}

/// A subject with 40 informative coordinates is kicked on 16 of them, the widest priors first.
/// `obj` is constant, so no kick is ever lower and `finish` never runs.
#[test]
fn a_kick_restart_visits_at_most_sixteen_coordinates_the_widest_first() {
    let sds: Vec<f64> = (0..40).map(|i| 1.0 + i as f64).collect();
    let mut visited = Vec::new();
    let x = vec![0.0; 40];
    let probe = |cand: &mut Vec<f64>| visited.push(cand.iter().position(|v| *v != 0.0).unwrap());
    let finish = |_: &mut Vec<f64>| panic!("obj is constant: no kick is ever lower");
    let r = kick_restarts(&x, 0.0, &sds, &|_: &[f64]| 1.0, probe, finish);
    assert!(r.is_none());
    assert_eq!(visited.len(), 32, "{visited:?}");
    assert!(visited.iter().all(|&j| j >= 24), "{visited:?}");
}

/// The escape's gate (#3210), all eight cells of analytic × `inner_kick` × kink flag: it runs on
/// an analytic gradient where the model has a kink or `inner_kick` asks for it, and never on an
/// FD gradient. Swapping either `&&`/`||`, or dropping any one term, flips at least one cell.
#[test]
fn the_stall_escape_runs_on_an_analytic_gradient_with_a_kink_or_inner_kick() {
    let mut model = crate::types::test_helpers::ode_model(crate::types::GradientMethod::Auto);
    // (analytic, inner_kick, kink) → runs.
    let table = [
        ((false, false, false), false),
        ((false, false, true), false),
        ((false, true, false), false),
        ((false, true, true), false),
        ((true, false, false), false),
        ((true, false, true), true),
        ((true, true, false), true),
        ((true, true, true), true),
    ];
    for ((analytic, kick, kink), want) in table {
        model.prediction_has_kink = kink;
        let got = kink_escape_allowed(&model, analytic, kick);
        assert_eq!(got, want, "analytic {analytic}, inner_kick {kick}, kink {kink}");
    }
}
