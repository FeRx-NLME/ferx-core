# Independent Laplace objective for the B1-shaped binary anchors (#2603), outside both
# engines. Pure Python, no dependencies.
#
#   python3 laplace_b1_check.py logit  profile_logit_b1.csv results/profile_logit_b1_laplace.ext
#   python3 laplace_b1_check.py probit probit_b1.csv        results/probit_b1_laplace.ext
#
# Reads the final row of a NONMEM .ext (32 THETA: TVL, EMAX, 30 free levels, each study's
# last visit dependent; OMEGA(1,1)), and prints -2 log L by Laplace per study:
#   f(eta)  = sum_r -log P(y_r | LP_r) + eta^2 / (2 omega)
#   -2 l_i  = 2 f(eta_hat) + log(omega * f''(eta_hat))
# with eta_hat found by Newton. NONMEM's METHOD=COND LAPLACE LIKE objective has this form
# (no 2*pi constant). Phi is math.erfc-based (correct to ~1 ulp), and -log Phi in the lower
# tail uses log(0.5 * erfc(-x / sqrt 2)) directly, which is accurate while erfc is
# representable (|x| < 26), far beyond this fixture's |LP| < 3.
import math, sys

link, data, ext = sys.argv[1], sys.argv[2], sys.argv[3]
names, row = None, None
for line in open(ext):
    f = line.split()
    if f and f[0] == "ITERATION":
        names = f
    elif f and f[0] == "-1000000000":
        row = dict(zip(names, map(float, f)))
th = [row[f"THETA{k}"] for k in range(1, 33)]
om = row["OMEGA(1,1)"]
tvl, emax = th[0], th[1]
times = [0.0, 4.0, 8.0, 12.0]


def placebo(s, t):
    lv = th[2 + 3 * (s - 1): 5 + 3 * (s - 1)]
    k = times.index(t)
    return lv[k] if k < 3 else -sum(lv)


SQRT2PI = math.sqrt(2 * math.pi)


def term(x):
    """g(x) = -log P(Y=1 | lp = x) for y = 1 (call with -x for y = 0), and g', g''."""
    if link == "logit":
        p = 1 / (1 + math.exp(-x))
        g = math.log1p(math.exp(-abs(x))) + max(-x, 0.0)
        return g, -(1 - p), p * (1 - p)
    cdf = 0.5 * math.erfc(-x / math.sqrt(2))
    pdf = math.exp(-0.5 * x * x) / SQRT2PI
    m = pdf / cdf
    g = -math.log(cdf) if x <= 0 else -math.log1p(-0.5 * math.erfc(x / math.sqrt(2)))
    return g, -m, m * (x + m)


by_study = {}
for i, line in enumerate(open(data)):
    if i == 0:
        continue
    f = line.strip().split(",")
    s, t, dv, trt = int(f[1]), float(f[2]), int(f[3]), float(f[7])
    by_study.setdefault(s, []).append((tvl + placebo(s, t) + emax * trt, 1.0 if dv == 1 else -1.0))

ofv = 0.0
for s, recs in sorted(by_study.items()):
    eta = 0.0
    for _ in range(100):
        g1, g2 = eta / om, 1 / om
        for lp, sg in recs:
            _, d1, d2 = term(sg * (lp + eta))
            g1 += sg * d1
            g2 += d2
        step = g1 / g2
        eta -= step
        if abs(step) < 1e-15:
            break
    fval, h = eta * eta / (2 * om), 1 / om
    for lp, sg in recs:
        g, _, d2 = term(sg * (lp + eta))
        fval += g
        h += d2
    ofv += 2 * fval + math.log(om * h)
print("%.13f  (NONMEM OBJ %.13f, diff %.3e)" % (ofv, row["OBJ"], ofv - row["OBJ"]))
