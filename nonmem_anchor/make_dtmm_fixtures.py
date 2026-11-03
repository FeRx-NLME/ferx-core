#!/usr/bin/env python3
"""Generate the #2473 discrete-time Markov (DTMM) anchor datasets.

Two mixed-effects DTMMs, each a chain over records where the probability of the state at
record k depends on the state observed at record k-1 (NONMEM's PREVIOUS-DV $PRED):

  dtmm_2state  awake (0) / asleep (1).
               P(0->1) = inv_logit(A_AS + B_DOSE*DOSE + ETA_AS)
               P(1->0) = inv_logit(A_SA + ETA_SA)
  dtmm_3state  ordered none (0) / mild (1) / severe (2), cumulative logit conditioned on
               the previous state j:
               P(s <= 0 | j) = inv_logit(A1 - G_j + ETA)
               P(s <= 1 | j) = inv_logit(A1 + exp(D2) - G_j + ETA),   G_0 = 0

Record spacing is deliberately irregular (a step is a record, not a time unit) and every
subject carries a few rows with a missing DV (MDV = 1): the chain then links to the last
observed state, which is what both engines score. DOSE is a time-constant covariate.

Columns: ID TIME DV MDV CMT DOSE PDV. PDV is the previous *observed* DV (-1 on a subject's
first observed record): NONMEM reads it, ferx ignores it and derives the same quantity from
the record order. MDV rows carry the PDV the next observed record will see, so NONMEM never
scores them. Stdlib only, seeded, so the files regenerate byte for byte:

    python3 make_dtmm_fixtures.py      # from nonmem_anchor/
"""

import math
import random


def inv_logit(x):
    return 1.0 / (1.0 + math.exp(-x))


def draw(probs, rng):
    u = rng.random()
    acc = 0.0
    for k, p in enumerate(probs):
        acc += p
        if u < acc:
            return k
    return len(probs) - 1


def times(rng, n):
    t, out = 0.0, []
    for _ in range(n):
        out.append(round(t, 2))
        t += rng.choice([0.5, 1.0, 1.0, 2.0, 3.5])
    return out


def write(name, rows):
    with open(name, "w") as f:
        f.write("ID,TIME,DV,MDV,CMT,DOSE,PDV\n")
        for r in rows:
            f.write(",".join(r) + "\n")


def chain(rng, n_subj, n_rec, row_probs, eta_sd, dose_levels):
    """row_probs(prev, dose, eta) -> list of next-state probabilities."""
    rows = []
    for i in range(1, n_subj + 1):
        dose = dose_levels[(i - 1) % len(dose_levels)]
        eta = [rng.gauss(0.0, sd) for sd in eta_sd]
        state = draw([0.6, 0.3, 0.1][: len(row_probs(0, dose, eta))], rng)
        prev_obs = -1
        for k, t in enumerate(times(rng, n_rec)):
            if k > 0:
                state = draw(row_probs(state, dose, eta), rng)
            missing = k > 0 and rng.random() < 0.08
            if missing:
                rows.append([str(i), "%g" % t, ".", "1", "5", "%g" % dose, str(prev_obs)])
            else:
                rows.append([str(i), "%g" % t, str(state), "0", "5", "%g" % dose, str(prev_obs)])
                prev_obs = state
    return rows


def two_state(prev, dose, eta):
    if prev == 0:
        p = inv_logit(-1.0 + 0.6 * dose + eta[0])
        return [1.0 - p, p]
    p = inv_logit(-0.7 + eta[1])
    return [p, 1.0 - p]


def three_state(prev, dose, eta):
    g = [0.0, 1.2, 2.4][prev]
    c0 = inv_logit(-0.3 - g + eta[0])
    c1 = inv_logit(-0.3 + math.exp(0.4) - g + eta[0])
    return [c0, c1 - c0, 1.0 - c1]


write("dtmm_2state.csv", chain(random.Random(2473), 60, 16, two_state, [0.6, 0.4], [0, 1, 2]))
write("dtmm_3state.csv", chain(random.Random(24730), 60, 14, three_state, [0.7], [0]))
