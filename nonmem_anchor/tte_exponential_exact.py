#!/usr/bin/env python3
"""Exact marginal likelihood of examples/tte_exponential.ferx, outside ferx and NONMEM (#1781).

Brute-force quadrature over eta, Newton to the minimum, then R^-1, S^-1 and R^-1 S R^-1 standard
errors there. Usage: python3 tte_exponential_exact.py data/tte_exponential.csv

With `--at LAMBDA OMEGA2` (repeatable) it also evaluates R^-1, S^-1 and R^-1 S R^-1 at those
fixed points, without moving them — the same-point anchor
src/estimation/covariance.rs::the_sampler_points_match_the_exact_marginal_covariance reads.
At a point that is not the minimum, R is the Hessian of the OFV in (log lambda, log omega2),
which is a linear map of ferx's packed (log lambda, log omega), so the two agree there too.

Precision: R is a central second difference of the summed OFV at h = 1e-4, S a central
difference of each subject's -log L at h = 1e-5. Against derivatives taken under the
integral instead (Codex review r2 on #1781), the printed SEs differ by up to ~2.5e-4
relative (SE(omega2), R and RSR); quote them to that precision, not to their printed digits.
"""
import csv
import math
import sys

# Composite Simpson rule on z = eta / omega over [-12, 12], 4001 points.
N_INTERVALS = 4000
Z_NODES = [-12 + 24 * k / N_INTERVALS for k in range(N_INTERVALS + 1)]
SIMPSON_WEIGHTS = [
    (1 if k in (0, N_INTERVALS) else (4 if k % 2 else 2)) * (24 / N_INTERVALS) / 3
    for k in range(N_INTERVALS + 1)
]
STD_NORMAL_PDF = [math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi) for z in Z_NODES]
HESSIAN_STEP = 1e-4
SCORE_STEP = 1e-5


def read_events(path):
    """(time, event flag) for every observation row of the dataset."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [(float(r["TIME"]), int(float(r["DV"]))) for r in rows if r["EVID"] == "0"]


def parse_points(args):
    """The `--at LAMBDA OMEGA2` pairs, in order."""
    points = []
    while args:
        if args[0] != "--at" or len(args) < 3:
            raise SystemExit(f"expected `--at LAMBDA OMEGA2`, got {args}")
        points.append((float(args[1]), float(args[2])))
        args = args[3:]
    return points


def subject_likelihood(lam, omega2, time, event):
    """L_i = integral of hazard^event * survival over eta ~ N(0, omega2)."""
    omega = math.sqrt(omega2)
    total = 0.0
    for z, weight, pdf in zip(Z_NODES, SIMPSON_WEIGHTS, STD_NORMAL_PDF):
        rate = lam * math.exp(omega * z)
        total += weight * pdf * (rate**event) * math.exp(-rate * time)
    return total


def ofv(data, x):
    """-2 log L at x = (log lambda, log omega2)."""
    lam, omega2 = math.exp(x[0]), math.exp(x[1])
    return -2 * sum(math.log(subject_likelihood(lam, omega2, t, d)) for t, d in data)


def subject_nll(x, time, event):
    """-log L_i at x = (log lambda, log omega2)."""
    return -math.log(subject_likelihood(math.exp(x[0]), math.exp(x[1]), time, event))


def shifted(x, steps):
    """x with `steps` (index -> increment) added."""
    out = list(x)
    for index, step in steps.items():
        out[index] += step
    return out


def gradient_and_hessian(func, x, h=HESSIAN_STEP):
    """Central-difference value, gradient and Hessian of `func` at `x`."""
    n = len(x)
    centre = func(x)
    grad = [0.0] * n
    hess = [[0.0] * n for _ in range(n)]
    for i in range(n):
        plus, minus = func(shifted(x, {i: h})), func(shifted(x, {i: -h}))
        grad[i] = (plus - minus) / (2 * h)
        hess[i][i] = (plus - 2 * centre + minus) / h**2
    for i in range(n):
        for j in range(i + 1, n):
            cross = (
                func(shifted(x, {i: h, j: h}))
                - func(shifted(x, {i: h, j: -h}))
                - func(shifted(x, {i: -h, j: h}))
                + func(shifted(x, {i: -h, j: -h}))
            )
            hess[i][j] = hess[j][i] = cross / (4 * h * h)
    return centre, grad, hess


def inverse2(matrix):
    """Inverse of a 2x2 matrix."""
    det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    return [
        [matrix[1][1] / det, -matrix[0][1] / det],
        [-matrix[1][0] / det, matrix[0][0] / det],
    ]


def matmul2(left, right):
    """Product of two 2x2 matrices."""
    return [[sum(left[i][k] * right[k][j] for k in range(2)) for j in range(2)] for i in range(2)]


def minimise(data, start):
    """Newton in (log lambda, log omega2) to |step| < 1e-9."""
    x = list(start)
    for _ in range(30):
        _, grad, hess = gradient_and_hessian(lambda v: ofv(data, v), x)
        det = hess[0][0] * hess[1][1] - hess[0][1] ** 2
        step = [
            -(hess[1][1] * grad[0] - hess[0][1] * grad[1]) / det,
            -(-hess[1][0] * grad[0] + hess[0][0] * grad[1]) / det,
        ]
        x = [x[0] + step[0], x[1] + step[1]]
        if max(abs(v) for v in step) < 1e-9:
            break
    return x


def score_cross_product(data, x):
    """S = sum over subjects of the outer product of central-difference -log L_i scores."""
    cross = [[0.0, 0.0], [0.0, 0.0]]
    for time, event in data:
        score = [
            (
                subject_nll(shifted(x, {k: SCORE_STEP}), time, event)
                - subject_nll(shifted(x, {k: -SCORE_STEP}), time, event)
            )
            / (2 * SCORE_STEP)
            for k in range(2)
        ]
        for a in range(2):
            for b in range(2):
                cross[a][b] += score[a] * score[b]
    return cross


def report(data, x, label):
    """Print SE(lambda), SE(omega2) under R^-1, S^-1 and R^-1 S R^-1 at x."""
    lam, omega2 = math.exp(x[0]), math.exp(x[1])
    _, _, hess = gradient_and_hessian(lambda v: ofv(data, v), x)
    cross = score_cross_product(data, x)
    # R is the Hessian of -log L, half the OFV Hessian.
    r_inv = inverse2([[hess[i][j] / 2 for j in range(2)] for i in range(2)])
    s_inv = inverse2(cross)
    sandwich = matmul2(matmul2(r_inv, cross), r_inv)
    print(f"{label} (lambda={lam}, omega2={omega2}):")
    for tag, cov in [("R", r_inv), ("S", s_inv), ("RSR", sandwich)]:
        print(
            f"  {tag:>3}: SE(lambda)={lam * math.sqrt(cov[0][0]):.7f} "
            f"SE(omega2)={omega2 * math.sqrt(cov[1][1]):.7f}"
        )


def print_reference_objectives(data):
    """Exact -2 log L at the estimates the #1781 PR compares."""
    for lam_at, omega2_at, tag in [
        (0.038756, 0.054346, "ferx SAEM"),
        (0.033848, 0.081064, "ferx IMP"),
        (0.033591, 0.082876, "ferx IMPMAP"),
        (0.0337025, 0.0904614, "NONMEM IMP"),
        (0.033183, 0.291718, "FOCEI min"),
    ]:
        at = [math.log(lam_at), math.log(omega2_at)]
        print(f"OFV_exact at {tag} ({lam_at}, {omega2_at}) = {ofv(data, at):.4f}")


def main():
    """Minimum, then the requested fixed points."""
    data = read_events(sys.argv[1])
    points = parse_points(sys.argv[2:])
    x_hat = minimise(data, [math.log(0.0337), math.log(0.08)])
    lam, omega2 = math.exp(x_hat[0]), math.exp(x_hat[1])
    value, grad, hess = gradient_and_hessian(lambda v: ofv(data, v), x_hat)
    print(f"MLE: lambda={lam:.6f} omega2={omega2:.6f} OFV={value:.4f} grad={grad}")
    det = hess[0][0] * hess[1][1] - hess[0][1] ** 2
    se_lambda = lam * math.sqrt(2 * hess[1][1] / det)
    se_omega2 = omega2 * math.sqrt(2 * hess[0][0] / det)
    print(f"R-inv SE: SE(lambda)={se_lambda:.6f} SE(omega2)={se_omega2:.6f}")
    print_reference_objectives(data)
    report(data, x_hat, "MLE")
    for lam_at, omega2_at in points:
        report(data, [math.log(lam_at), math.log(omega2_at)], "point")


if __name__ == "__main__":
    main()
