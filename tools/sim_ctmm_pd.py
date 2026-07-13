#!/usr/bin/env python3
"""Simulate the drug-driven (time-inhomogeneous) 2-state CTMM in examples/ctmm_pd_2state.ferx.

A 1-compartment IV bolus drives the concentration C(t) = (D/V)·exp(-(CL/V)·t); the
s0 -> s1 intensity is q01(t) = exp(LQ01 + SLOPE·C(t)) and s1 -> s0 is q10 = exp(LQ10)
(concentration-independent). Between-subject variability is a log-normal random effect
on CL. The continuous-time chain is simulated on a fine Euler-Bernoulli grid (the rate is
time-varying, so this is a non-homogeneous Poisson process) and the state is sampled at a
fixed observation schedule. Deterministic given SEED, so the committed CSV is reproducible.

Writes data/ctmm_pd_2state.csv (columns ID,TIME,DV,EVID,AMT,CMT,MDV): one IV dose row per
subject (EVID=1, CMT=1) plus the state observations (EVID=0, CMT=5, DV in {0,1}).
"""
import math
import random

SEED = 20817
N_SUBJECTS = 60
DOSE = 100.0
TVCL, TVV = 1.0, 10.0
OMEGA_CL = 0.09  # variance of ETA_CL
LQ01, LQ10, SLOPE = -1.2, -0.7, 0.15
OBS_TIMES = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0]
GRID_DT = 0.005  # fine step for the non-homogeneous chain


def conc(t, cl, v):
    return (DOSE / v) * math.exp(-(cl / v) * t)


def simulate_subject(rng):
    eta = rng.gauss(0.0, math.sqrt(OMEGA_CL))
    cl = TVCL * math.exp(eta)
    v = TVV
    q10 = math.exp(LQ10)

    state = 0  # everyone starts in s0
    t = 0.0
    t_end = OBS_TIMES[-1]
    obs = {}
    obs_idx = 0
    n_steps = int(round(t_end / GRID_DT))
    for step in range(n_steps + 1):
        t = step * GRID_DT
        # Record any observation times reached in this step.
        while obs_idx < len(OBS_TIMES) and OBS_TIMES[obs_idx] <= t + 1e-9:
            obs[OBS_TIMES[obs_idx]] = state
            obs_idx += 1
        # Transition over (t, t+dt] with the current-state rate at time t.
        if state == 0:
            rate = math.exp(LQ01 + SLOPE * conc(t, cl, v))
        else:
            rate = q10
        if rng.random() < 1.0 - math.exp(-rate * GRID_DT):
            state = 1 - state
    return obs


def main():
    rng = random.Random(SEED)
    rows = ["ID,TIME,DV,EVID,AMT,CMT,MDV"]
    for sid in range(1, N_SUBJECTS + 1):
        rows.append(f"{sid},0,0,1,{DOSE:g},1,1")  # IV bolus into central (cmt 1)
        obs = simulate_subject(rng)
        for t in OBS_TIMES:
            rows.append(f"{sid},{t:g},{obs[t]},0,0,5,0")  # state observation on cmt 5
    out = "data/ctmm_pd_2state.csv"
    with open(out, "w") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"wrote {out}: {N_SUBJECTS} subjects, {len(OBS_TIMES)} obs each")


if __name__ == "__main__":
    main()
