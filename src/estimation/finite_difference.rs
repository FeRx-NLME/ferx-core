//! Bounded, noise-aware scalar finite differences used by outer optimizers.
//!
//! This module deliberately operates in the caller's coordinates.  It knows
//! nothing about model parameters or penalties: an evaluator must report an
//! unusable objective value as `None`.

use crate::types::OuterFdMethod;

const MAX_SEARCH_ITERS: usize = 15;
const RATIO_LOW: f64 = 1.5;
const RATIO_HIGH: f64 = 6.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct AxisBounds {
    pub lower: f64,
    pub upper: f64,
}

impl AxisBounds {
    fn central_cap(self, x: f64) -> f64 {
        ((x - self.lower).min(self.upper - x) / 3.0).max(0.0)
    }

    fn forward_cap(self, x: f64) -> f64 {
        ((self.upper - x) / 2.0).max(0.0)
    }
}

/// Select a first-derivative interval and evaluate its derivative.
///
/// `noise` is an absolute bound in the evaluator's output units.  Adaptive
/// methods return `None` if no complete, representable stencil fits inside the
/// supplied bounds.  Fixed differences retain the historical centred/clamped
/// convention in their caller.
pub(crate) fn adaptive_first_derivative(
    method: OuterFdMethod,
    x: f64,
    initial_h: f64,
    bounds: AxisBounds,
    noise: f64,
    mut eval: impl FnMut(f64) -> Option<f64>,
) -> Option<f64> {
    match method {
        OuterFdMethod::Fixed => None,
        OuterFdMethod::Shi => shi_central(x, initial_h, bounds, noise, eval),
        OuterFdMethod::Gill => gill_forward(x, initial_h, bounds, noise, eval),
    }
}

/// Shi's centred search applied to one column of a vector-valued prediction.
/// A single interval is accepted only when every finite component's ratio is
/// in range; the largest component ratio drives the conservative search.
pub(crate) fn shi_central_vector_derivative(
    x: f64,
    initial_h: f64,
    bounds: AxisBounds,
    noise: f64,
    mut eval: impl FnMut(f64) -> Option<Vec<f64>>,
) -> Option<Vec<f64>> {
    if !usable_h(noise) {
        return None;
    }
    let cap = bounds.central_cap(x);
    let mut h = initial_h.min(cap);
    let mut last = None;
    for _ in 0..MAX_SEARCH_ITERS {
        if !usable_h(h) || h > cap || x + 3.0 * h > bounds.upper || x - 3.0 * h < bounds.lower {
            break;
        }
        let fp = eval(x + h)?;
        let fm = eval(x - h)?;
        let fp3 = eval(x + 3.0 * h)?;
        let fm3 = eval(x - 3.0 * h)?;
        if fp.len() != fm.len() || fp.len() != fp3.len() || fp.len() != fm3.len() {
            return None;
        }
        let mut derivative = Vec::with_capacity(fp.len());
        let mut ratio = 0.0_f64;
        for i in 0..fp.len() {
            let d = (fp[i] - fm[i]) / (2.0 * h);
            let r = (fp3[i] - 3.0 * fp[i] + 3.0 * fm[i] - fm3[i]).abs() / (8.0 * noise);
            if !d.is_finite() || !r.is_finite() {
                return last;
            }
            derivative.push(d);
            ratio = ratio.max(r);
        }
        last = Some(derivative);
        if (RATIO_LOW..=RATIO_HIGH).contains(&ratio) {
            return last;
        }
        let next = if ratio < RATIO_LOW {
            (2.0 * h).min(cap)
        } else {
            h / 2.0
        };
        if next == h || !usable_h(next) {
            break;
        }
        h = next;
    }
    last
}

fn usable_h(h: f64) -> bool {
    h.is_finite() && h > 0.0
}

fn shi_central(
    x: f64,
    initial_h: f64,
    bounds: AxisBounds,
    noise: f64,
    mut eval: impl FnMut(f64) -> Option<f64>,
) -> Option<f64> {
    if !usable_h(noise) {
        return None;
    }
    let cap = bounds.central_cap(x);
    let mut h = initial_h.min(cap);
    if !usable_h(h) || x + 3.0 * h == x || x - 3.0 * h == x {
        return None;
    }
    let mut last = None;
    for _ in 0..MAX_SEARCH_ITERS {
        if !usable_h(h) || h > cap || x + 3.0 * h > bounds.upper || x - 3.0 * h < bounds.lower {
            break;
        }
        let fp = eval(x + h)?;
        let fm = eval(x - h)?;
        let fp3 = eval(x + 3.0 * h)?;
        let fm3 = eval(x - 3.0 * h)?;
        let derivative = (fp - fm) / (2.0 * h);
        if !derivative.is_finite() {
            return last;
        }
        last = Some(derivative);
        let ratio = (fp3 - 3.0 * fp + 3.0 * fm - fm3).abs() / (8.0 * noise);
        if !ratio.is_finite() || (RATIO_LOW..=RATIO_HIGH).contains(&ratio) {
            return last;
        }
        let next = if ratio < RATIO_LOW {
            (2.0 * h).min(cap)
        } else {
            h / 2.0
        };
        if next == h || !usable_h(next) {
            break;
        }
        h = next;
    }
    last
}

// A bounded forward-interval implementation of the elementary Gill et al.
// curvature/noise balance.  It is intentionally separate from Shi: its chosen
// interval is used only with the forward stencil, never re-used as a central
// interval.  The full Gill classification search remains future work.
fn gill_forward(
    x: f64,
    initial_h: f64,
    bounds: AxisBounds,
    noise: f64,
    mut eval: impl FnMut(f64) -> Option<f64>,
) -> Option<f64> {
    if !usable_h(noise) {
        return None;
    }
    let cap = bounds.forward_cap(x);
    let h0 = initial_h.min(cap);
    if !usable_h(h0) || x + 2.0 * h0 == x {
        return None;
    }
    let f0 = eval(x)?;
    let f1 = eval(x + h0)?;
    let f2 = eval(x + 2.0 * h0)?;
    let curvature = ((f2 - 2.0 * f1 + f0) / (h0 * h0)).abs();
    let h = if curvature.is_finite() && curvature > 0.0 {
        (2.0 * (noise / curvature).sqrt()).min(cap)
    } else {
        h0
    };
    if !usable_h(h) || x + h == x || x + h > bounds.upper {
        return None;
    }
    let fh = eval(x + h)?;
    let derivative = (fh - f0) / h;
    derivative.is_finite().then_some(derivative)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOUNDS: AxisBounds = AxisBounds {
        lower: -10.0,
        upper: 10.0,
    };

    #[test]
    fn shi_recovers_cubic_derivative() {
        let d = adaptive_first_derivative(OuterFdMethod::Shi, 2.0, 0.01, BOUNDS, 1e-8, |x| {
            Some(x * x * x)
        });
        assert!((d.unwrap() - 12.0).abs() < 1e-3);
    }

    #[test]
    fn shi_does_not_probe_past_bound() {
        let d = adaptive_first_derivative(
            OuterFdMethod::Shi,
            0.99,
            0.1,
            AxisBounds {
                lower: -1.0,
                upper: 1.0,
            },
            1e-8,
            |x| (x <= 1.0 && x >= -1.0).then_some(x * x),
        );
        assert!(d.is_some());
    }

    #[test]
    fn gill_uses_forward_derivative() {
        let d = adaptive_first_derivative(OuterFdMethod::Gill, 1.0, 0.01, BOUNDS, 1e-8, |x| {
            Some(x * x)
        });
        assert!((d.unwrap() - 2.0).abs() < 1e-3);
    }

    #[test]
    fn vector_shi_recovers_component_derivatives() {
        let d = shi_central_vector_derivative(2.0, 0.01, BOUNDS, 1e-8, |x| {
            Some(vec![x * x, x * x * x])
        })
        .unwrap();
        assert!((d[0] - 4.0).abs() < 1e-3);
        assert!((d[1] - 12.0).abs() < 1e-3);
    }
}
