# Committed with #2603 as the reference of `src/categorical/probit_tests.rs`:
#   python3 probit_mills_reference.py -40 -2 0 4 30
# High-precision reference for g(x) = -log Phi(x) and its first four derivatives (#2603).
# Pure Python `decimal` at 80 digits: Phi from the erf Taylor series (|x| <= 3) or the
# Mills-ratio continued fraction (|x| > 3); derivatives from the exact recursions in (x, m)
# evaluated at 80 digits (no cancellation at that precision), cross-checked by a
# high-precision central difference of g.
from decimal import Decimal as D, getcontext
import sys
getcontext().prec = 80
PI = D("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899")
SQRT2PI = (2 * PI).sqrt()

def phi_pdf(x):
    return (-(x * x) / 2).exp() / SQRT2PI

def Phi(x):
    if abs(x) <= 3:
        # erf(z) = 2/sqrt(pi) * sum (-1)^n z^(2n+1) / (n! (2n+1))
        z = x / D(2).sqrt()
        s, term, n = D(0), z, 0
        while True:
            add = term / (2 * n + 1)
            s += add
            if abs(add) < D(10) ** -85:
                break
            n += 1
            term = -term * z * z / n
        erf = 2 / PI.sqrt() * s
        return (1 + erf) / 2
    t = abs(x)
    # Mills ratio R(t) = Q(t)/phi(t) by the continued fraction 1/(t+1/(t+2/(t+3/(t+...))))
    f = t
    for k in range(4000, 0, -1):
        f = t + k / f
    q = phi_pdf(t) / f
    return q if x < 0 else 1 - q

def jet(x):
    x = D(x)
    m = phi_pdf(x) / Phi(x)
    a = x + m
    m1 = -m * a
    m2 = -m1 * a - m * (1 + m1)
    m3 = -m2 * a - 2 * m1 * (1 + m1) - m * m2
    if x > 3:
        # Upper tail: -log(1 - Q) from Q itself, since 1 - Q rounds to 1 at 80 digits.
        t = x
        f = t
        for k in range(4000, 0, -1):
            f = t + k / f
        q = phi_pdf(t) / f
        val = q + q * q / 2 + q ** 3 / 3  # -log(1 - q)
    else:
        val = -Phi(x).ln()
    return [val, -m, -m1, -m2, -m3]

if __name__ == "__main__":
    # FD self-check of the recursion at a few points.
    h = D(10) ** -20
    for x in ["-3.5", "0.3", "2.5", "-12"]:
        j = jet(x)
        for k in range(4):
            fd = (jet(D(x) + h)[k] - jet(D(x) - h)[k]) / (2 * h)
            rel = abs(fd - j[k + 1]) / max(abs(j[k + 1]), D(10) ** -60)
            assert rel < D(10) ** -25, (x, k, rel)
    for x in sys.argv[1:]:
        print(x, " ".join("%.20e" % v for v in jet(x)))
