//! The probit record term (#2603): `−log Φ(s·lp)`, its four derivatives, and the probability
//! the sampler draws from.
use super::*;
use crate::sens::taylor::Taylor;

/// `[x, g, g′, g″, g‴, g⁗]` for `g(x) = −log Φ(x)`, from an 80-digit reference computed
/// outside ferx (pure-Python `decimal`: `Φ` by the erf series for `|x| ≤ 3` and the Mills
/// continued fraction beyond, the derivatives by the same recursion at 80 digits, which is
/// cross-checked there against an 80-digit central difference of `g`). Script: `nonmem_anchor/probit_mills_reference.py`.
/// `x = 40`'s derivatives underflow to `0` in `f64`, as they should.
// The x = 0 row's value and second derivative are `ln 2` and `2/π` exactly (−log Φ(0) = ln 2,
// g″(0) = 2/π), spelled as the constants; every other entry keeps the reference's 21 digits.
#[allow(clippy::excessive_precision)]
const REFERENCE: [[f64; 6]; 23] = [
    [
        -100.0,
        5.00552420869420529925e+03,
        -1.00009998000999260626e+02,
        9.99900059950051711688e-01,
        -1.99760299586234317927e-06,
        -5.98802096277375412521e-08,
    ],
    [
        -20.0,
        2.03917155371097265970e+02,
        -2.00497530685278491092e+01,
        9.97536738384947829772e-01,
        -2.42726578935842021842e-04,
        -3.57035515884565913809e-05,
    ],
    [
        -3.0,
        6.60772622151034916271e+00,
        -3.28309865493043639972e+00,
        9.29440813214731931247e-01,
        -3.14706728308424901752e-02,
        -1.89305471020773767588e-02,
    ],
    [
        -2.01,
        3.80696078493148171162e+00,
        -2.38207570305353977247e+00,
        8.86312491940400848378e-01,
        -5.89630928743846843831e-02,
        -3.91320319931866417162e-02,
    ],
    [
        -2.0,
        3.78318433368203210776e+00,
        -2.37321553282284103403e+00,
        8.85720899585918708397e-01,
        -5.93558612915658159737e-02,
        -3.94219938659468166486e-02,
    ],
    [
        -1.99,
        3.75949645448968938766e+00,
        -2.36436129820252016742e+00,
        8.85125365014891807292e-01,
        -5.97515396150234065487e-02,
        -3.97140147306371954494e-02,
    ],
    [
        2.0,
        2.30129093289634899244e-02,
        -5.52478626789899562866e-02,
        1.13548051688576442930e-01,
        -1.84394815032477593642e-01,
        1.87854685611609700802e-01,
    ],
    [
        15.0,
        3.67096619931275082910e-51,
        -5.53070954984441638358e-50,
        8.29606432476662476528e-49,
        -1.23887893916514933069e-47,
        1.84172628009819069182e-46,
    ],
    [
        -1000.0,
        5.00007826694812159985e+05,
        -1.00000099999799999750e+03,
        9.99999000005999949536e-01,
        -1.99997600029999599297e-09,
        -5.99988000209996244870e-12,
    ],
    [
        -40.0,
        8.04608442013753801803e+02,
        -4.00249688472072620016e+01,
        9.99377331621408626638e-01,
        -3.10174403964862512376e-05,
        -2.31477004389180680927e-06,
    ],
    [
        -37.5,
        7.07668989317507225678e+02,
        -3.75266288748836558398e+01,
        9.99291905114579215308e-01,
        -3.76051401563817362900e-05,
        -2.99145309924538489689e-06,
    ],
    [
        -37.01,
        6.89400905417394483266e+02,
        -3.70369804153025441451e+01,
        9.99273113157040371313e-01,
        -3.91098268039374226423e-05,
        -3.15186820153864646720e-06,
    ],
    [
        -36.99,
        6.88660365663658922131e+02,
        -3.70169949608655741713e+01,
        9.99272330329679325800e-01,
        -3.91729318664752303517e-05,
        -3.15864107666498571974e-06,
    ],
    [
        -30.0,
        4.54321243956343209902e+02,
        -3.00332596674336755882e+01,
        9.98896228488109882981e-01,
        -7.30999301578443829927e-05,
        -7.24593721098034199238e-06,
    ],
    [
        -10.0,
        5.32312851505124697837e+01,
        -1.00980932339625120164e+01,
        9.90554622174343757379e-01,
        -1.78640039211650685294e-03,
        -4.97853822379440186414e-04,
    ],
    [
        -5.0,
        1.50649983939887253115e+01,
        -5.18650396712584171866e+00,
        9.67303565382887753366e-01,
        -1.08257645063566994986e-02,
        -5.08783697388744647616e-03,
    ],
    [
        -1.0,
        1.84102164500926357427e+00,
        -1.52513527616098110151e+00,
        8.00902334429651197034e-01,
        -1.16931195406048835106e-01,
        -7.91749836807456308652e-02,
    ],
    [
        0.0,
        std::f64::consts::LN_2,
        -7.97884560802865405726e-01,
        std::f64::consts::FRAC_2_PI,
        -2.18013614144990153232e-01,
        -1.14770682054218864376e-01,
    ],
    [
        0.7,
        2.77023942277131229872e-01,
        -4.11924750419290663928e-01,
        4.58029325301498368983e-01,
        -2.86043008310829793839e-01,
        -6.05904294978809807026e-02,
    ],
    [
        4.0,
        3.16717433774890095373e-05,
        -1.33834464468575138965e-04,
        5.35355769538180194418e-04,
        -2.00773191178957831349e-03,
        6.96132672713232949352e-03,
    ],
    [
        8.0,
        6.22096057427178584651e-16,
        -5.05227108353689509902e-15,
        4.04181686682951860357e-14,
        -3.18293078262825041259e-13,
        2.46550828876601636987e-12,
    ],
    [
        30.0,
        4.90671392714818717732e-198,
        -1.47364613487854759907e-196,
        4.42093840463564222687e-195,
        -1.32480787525581413662e-193,
        3.96558174895817109347e-192,
    ],
    [40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
];

/// Relative error, with exact zeros compared exactly.
fn rel(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        got.abs()
    } else {
        ((got - want) / want).abs()
    }
}

/// The value and every derivative of `−log Φ(x)` against the 80-digit reference, at every
/// order and in both tails, including where `Φ` underflows (x ≤ −38) and where it rounds to
/// `1` (x ≥ 9). The worst realised relative error per order is printed and bounds it.
///
/// Mutations it exists to catch: the value taken as `−ln(normal_cdf(x))` (`+∞` below −38);
/// `m` taken as `φ/Φ` directly (`NaN` below −38); any one of the recursions for `m″`, `m‴`
/// altered.
#[test]
fn probit_jet_matches_the_80_digit_reference_in_both_tails() {
    let mut worst = [0.0f64; 5];
    for row in REFERENCE {
        let got = neg_log_normal_cdf_jet(row[0]);
        for k in 0..5 {
            assert!(
                got[k].is_finite(),
                "x = {}: order {k} is {} (want {})",
                row[0],
                got[k],
                row[k + 1]
            );
            let e = rel(got[k], row[k + 1]);
            assert!(e.is_finite(), "x = {}: order {k} error {e}", row[0]);
            worst[k] = worst[k].max(e);
        }
    }
    eprintln!("probit jet vs 80-digit reference, worst relative error per order: {worst:?}");
    // Measured (macOS arm64), worst per order: 4.9e-14, 2.4e-15, 1.8e-14, 8.3e-13, 4.1e-12;
    // the bounds keep ~5× headroom. A direct recursion in `m = φ/Φ` gave 1.0 on the
    // value (the upper tail rounded to 0) and 5e6 / 1.7e12 on the third / fourth orders (the
    // far lower tail cancelled) — the two defects the tail forms exist to remove.
    let bound = [2.5e-13, 1.5e-14, 1e-13, 4e-12, 2e-11];
    for k in 0..5 {
        assert!(worst[k] < bound[k], "order {k}: {:.2e}", worst[k]);
    }
}

/// The record term on a fourth-order Taylor jet: the value and `d^k/dlp^k` of
/// `binary_record_nll_g(Probit, lp, y)` are `s^k g^(k)(s·lp)` with `s = 2y − 1` — both
/// outcomes, at points in each tail and in the middle — and the value is `−log P(y)` computed
/// from `Φ` directly wherever `Φ` is representable.
///
/// Mutations: the sign `s` dropped (y = 0 scored as y = 1); `s` applied to the even orders
/// too; any lifted order swapped for the logit arm's.
#[test]
fn probit_record_term_carries_s_to_the_k_on_every_order() {
    type T4 = Taylor<1, 4, 5>;
    let fact = [1.0, 1.0, 2.0, 6.0, 24.0];
    for lp in [-6.0, -1.3, 0.0, 0.7, 4.2, -45.0, 45.0] {
        for y in [0.0, 1.0] {
            let s = if y > 0.5 { 1.0 } else { -1.0 };
            let g = neg_log_normal_cdf_jet(s * lp);
            let t = binary_record_nll_g(LinkFn::Probit, T4::seed(lp, 0), y);
            let f = binary_record_nll_g(LinkFn::Probit, lp, y);
            assert_eq!(
                t.c[0], f,
                "lp {lp} y {y}: the jet's value is the f64 kernel's"
            );
            for k in 0..5 {
                let want = s.powi(k as i32) * g[k];
                let got = t.c[k] * fact[k];
                assert!(got.is_finite(), "lp {lp} y {y} order {k}: {got}");
                assert!(
                    (got - want).abs() <= 1e-14 * want.abs().max(1e-300),
                    "lp {lp} y {y} order {k}: {got} vs {want}"
                );
            }
            // Against Φ itself where it is representable and not 1.
            let p = crate::stats::special::normal_cdf(lp);
            if p > 1e-300 && p < 1.0 - 1e-12 {
                let direct = if y > 0.5 { -p.ln() } else { -(1.0 - p).ln() };
                assert!(
                    (f - direct).abs() < 1e-11 * direct.abs().max(1.0),
                    "lp {lp} y {y}: {f} vs −log P(y) = {direct}"
                );
            }
        }
    }
}

/// The jet's derivatives against central differences of the `f64` record term, which share
/// no derivative code with it, at moderate points where a difference is accurate.
#[test]
fn probit_record_term_derivatives_match_fd_of_the_value() {
    type T4 = Taylor<1, 4, 5>;
    for lp in [-2.5, -0.4, 0.0, 0.9, 2.2] {
        for y in [0.0, 1.0] {
            let f = |x: f64| binary_record_nll_g(LinkFn::Probit, x, y);
            let t = binary_record_nll_g(LinkFn::Probit, T4::seed(lp, 0), y);
            let h = 1e-4;
            let d1 = (f(lp + h) - f(lp - h)) / (2.0 * h);
            let d2 = (f(lp + h) - 2.0 * f(lp) + f(lp - h)) / (h * h);
            // Third and fourth: differences of the jet's own lower orders, one step up.
            let j = |x: f64| binary_record_nll_g(LinkFn::Probit, T4::seed(x, 0), y);
            let d3 = (j(lp + h).c[2] - j(lp - h).c[2]) * 2.0 / (2.0 * h);
            let d4 = (j(lp + h).c[3] - j(lp - h).c[3]) * 6.0 / (2.0 * h);
            for (got, fd, what, tol) in [
                (t.c[1], d1, "d1", 1e-8),
                (t.c[2] * 2.0, d2, "d2", 1e-5),
                (t.c[3] * 6.0, d3, "d3", 1e-7),
                (t.c[4] * 24.0, d4, "d4", 1e-7),
            ] {
                assert!(got.is_finite() && fd.is_finite(), "lp {lp} y {y} {what}");
                assert!(
                    (got - fd).abs() < tol * fd.abs().max(1.0),
                    "lp {lp} y {y} {what}: {got} vs FD {fd}"
                );
            }
        }
    }
}

/// The sampler's `p` and the likelihood describe one Bernoulli law under probit, as
/// `sampler_probability_agrees_with_likelihood` pins for logit: `−log P(y)` from
/// `binary_prob` equals `binary_data_term`. Mutation: `binary_prob`'s probit arm returning
/// `inv_logit`.
#[test]
fn probit_sampler_probability_agrees_with_likelihood() {
    let cov = HashMap::new();
    let lp_fn = |v: f64| -> LinearPredictorFn {
        Box::new(move |_: &[f64], _: &[f64], _: &HashMap<String, f64>| v)
    };
    for lp in [-3.0_f64, -0.8, 0.0, 0.6, 2.4] {
        let p = binary_prob(LinkFn::Probit, lp);
        for state in [0usize, 1] {
            let rec = ObsRecord::DiscreteState {
                time: 0.0,
                raw_time: 0.0,
                state,
                cmt: 3,
            };
            let from_p = if state == 1 { -p.ln() } else { -(1.0 - p).ln() };
            let from_term =
                binary_data_term(LinkFn::Probit, &lp_fn(lp), 3, &[rec], &[], &[], None, &cov);
            assert!(
                (from_p - from_term).abs() < 1e-11,
                "lp {lp} y {state}: sampler p implies {from_p}, likelihood {from_term}"
            );
        }
        // Probit and logit are different laws away from lp = 0.
        if lp != 0.0 {
            assert!((p - binary_prob(LinkFn::Logit, lp)).abs() > 1e-3, "lp {lp}");
        }
    }
}
