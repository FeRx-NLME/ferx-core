/// Forward-mode dual number: val + deriv * ε
///
/// Used for automatic differentiation through the expression evaluator
/// and PK model equations. Each operation propagates the derivative
/// via the chain rule.
#[derive(Debug, Clone, Copy)]
pub struct Dual {
    pub val: f64,
    pub deriv: f64,
}

impl Dual {
    pub fn new(val: f64, deriv: f64) -> Self {
        Self { val, deriv }
    }

    pub fn constant(val: f64) -> Self {
        Self { val, deriv: 0.0 }
    }

    pub fn variable(val: f64) -> Self {
        Self { val, deriv: 1.0 }
    }

    pub fn exp(self) -> Self {
        let e = self.val.exp();
        Self {
            val: e,
            deriv: self.deriv * e,
        }
    }

    pub fn ln(self) -> Self {
        Self {
            val: self.val.max(1e-30).ln(),
            deriv: self.deriv / self.val.max(1e-30),
        }
    }

    pub fn sqrt(self) -> Self {
        let s = self.val.max(0.0).sqrt();
        Self {
            val: s,
            deriv: if s > 1e-30 {
                self.deriv / (2.0 * s)
            } else {
                0.0
            },
        }
    }

    pub fn abs(self) -> Self {
        if self.val >= 0.0 {
            self
        } else {
            Self {
                val: -self.val,
                deriv: -self.deriv,
            }
        }
    }

    pub fn powf(self, exp: Dual) -> Self {
        if self.val <= 0.0 {
            return Self::constant(0.0);
        }
        let val = self.val.powf(exp.val);
        // d/dx (x^y) = y*x^(y-1)*dx + x^y*ln(x)*dy
        let deriv = val * (exp.val * self.deriv / self.val + exp.deriv * self.val.ln());
        Self { val, deriv }
    }

    pub fn powi(self, n: i32) -> Self {
        let val = self.val.powi(n);
        let deriv = self.deriv * n as f64 * self.val.powi(n - 1);
        Self { val, deriv }
    }

    pub fn max(self, other: f64) -> Self {
        if self.val >= other {
            self
        } else {
            Self::constant(other)
        }
    }
}

impl std::ops::Add for Dual {
    type Output = Dual;
    fn add(self, rhs: Dual) -> Dual {
        Dual {
            val: self.val + rhs.val,
            deriv: self.deriv + rhs.deriv,
        }
    }
}

impl std::ops::Sub for Dual {
    type Output = Dual;
    fn sub(self, rhs: Dual) -> Dual {
        Dual {
            val: self.val - rhs.val,
            deriv: self.deriv - rhs.deriv,
        }
    }
}

impl std::ops::Mul for Dual {
    type Output = Dual;
    fn mul(self, rhs: Dual) -> Dual {
        Dual {
            val: self.val * rhs.val,
            deriv: self.val * rhs.deriv + self.deriv * rhs.val,
        }
    }
}

impl std::ops::Div for Dual {
    type Output = Dual;
    fn div(self, rhs: Dual) -> Dual {
        if rhs.val.abs() < 1e-30 {
            return Dual::constant(0.0);
        }
        Dual {
            val: self.val / rhs.val,
            deriv: (self.deriv * rhs.val - self.val * rhs.deriv) / (rhs.val * rhs.val),
        }
    }
}

impl std::ops::Neg for Dual {
    type Output = Dual;
    fn neg(self) -> Dual {
        Dual {
            val: -self.val,
            deriv: -self.deriv,
        }
    }
}

impl PartialEq for Dual {
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
    }
}

impl PartialOrd for Dual {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

#[cfg(test)]
mod tests {
    use super::Dual;

    const TOL: f64 = 1e-9;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < TOL
    }

    #[test]
    fn constructors_set_value_and_seed_derivative() {
        assert_eq!(Dual::new(2.0, 3.0).deriv, 3.0);
        assert_eq!(Dual::constant(2.0).deriv, 0.0);
        assert_eq!(Dual::variable(2.0).deriv, 1.0);
    }

    #[test]
    fn exp_ln_sqrt_derivatives() {
        // d/dx e^x at x=1 → e, e.
        let e = Dual::variable(1.0).exp();
        assert!(close(e.val, std::f64::consts::E) && close(e.deriv, std::f64::consts::E));
        // d/dx ln(x) at x=2 → ln2, 0.5.
        let l = Dual::variable(2.0).ln();
        assert!(close(l.val, 2.0_f64.ln()) && close(l.deriv, 0.5));
        // d/dx sqrt(x) at x=4 → 2, 0.25.
        let s = Dual::variable(4.0).sqrt();
        assert!(close(s.val, 2.0) && close(s.deriv, 0.25));
        // sqrt at 0 → value 0, derivative floored to 0 (no divide-by-zero).
        let s0 = Dual::variable(0.0).sqrt();
        assert!(close(s0.val, 0.0) && close(s0.deriv, 0.0));
        // ln floors the argument at 1e-30 for non-positive input.
        let lneg = Dual::variable(-1.0).ln();
        assert!(lneg.val.is_finite());
    }

    #[test]
    fn abs_branches() {
        let pos = Dual::new(3.0, 1.0).abs();
        assert!(close(pos.val, 3.0) && close(pos.deriv, 1.0));
        let neg = Dual::new(-3.0, 1.0).abs();
        assert!(close(neg.val, 3.0) && close(neg.deriv, -1.0));
    }

    #[test]
    fn powf_and_powi() {
        // d/dx x^3 at x=2 → 8, 12 (powf with constant exponent).
        let p = Dual::variable(2.0).powf(Dual::constant(3.0));
        assert!(close(p.val, 8.0) && close(p.deriv, 12.0));
        // non-positive base short-circuits to constant 0.
        let p0 = Dual::variable(-2.0).powf(Dual::constant(2.0));
        assert!(close(p0.val, 0.0) && close(p0.deriv, 0.0));
        // powi: d/dx x^3 at x=2 → 8, 12.
        let pi = Dual::variable(2.0).powi(3);
        assert!(close(pi.val, 8.0) && close(pi.deriv, 12.0));
    }

    #[test]
    fn max_picks_branch_and_kills_derivative_when_clamped() {
        let kept = Dual::new(7.0, 1.0).max(5.0);
        assert!(close(kept.val, 7.0) && close(kept.deriv, 1.0));
        let clamped = Dual::new(2.0, 1.0).max(5.0);
        assert!(close(clamped.val, 5.0) && close(clamped.deriv, 0.0));
    }

    #[test]
    fn arithmetic_follows_calculus_rules() {
        let x = Dual::variable(3.0); // val 3, deriv 1
        let y = Dual::new(2.0, 0.0); // val 2, deriv 0
        let add = x + y;
        assert!(close(add.val, 5.0) && close(add.deriv, 1.0));
        let sub = x - y;
        assert!(close(sub.val, 1.0) && close(sub.deriv, 1.0));
        // product rule: d(xy) = x'y + xy' = 1*2 + 3*0 = 2.
        let mul = x * y;
        assert!(close(mul.val, 6.0) && close(mul.deriv, 2.0));
        // quotient rule: d(x/y) = (x'y - xy')/y² = (1*2 - 3*0)/4 = 0.5.
        let div = x / y;
        assert!(close(div.val, 1.5) && close(div.deriv, 0.5));
        // divide-by-(near)zero short-circuits to constant 0.
        let div0 = x / Dual::constant(0.0);
        assert!(close(div0.val, 0.0) && close(div0.deriv, 0.0));
        // negation flips both components.
        let neg = -x;
        assert!(close(neg.val, -3.0) && close(neg.deriv, -1.0));
    }

    #[test]
    fn eq_and_ord_compare_value_only() {
        // Equality ignores the derivative component.
        assert_eq!(Dual::new(2.0, 1.0), Dual::new(2.0, 9.0));
        assert_ne!(Dual::new(2.0, 1.0), Dual::new(3.0, 1.0));
        assert!(Dual::new(1.0, 5.0) < Dual::new(2.0, 0.0));
        assert!(Dual::new(2.0, 0.0) > Dual::new(1.0, 5.0));
    }
}
