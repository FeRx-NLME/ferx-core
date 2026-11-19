"""Dataset for `moving_infusion_end.ctl` / `moving_infusion_end_fit.ferx` (#3010).

A 1-compartment disposition dosed by duration-modeled infusions (`RATE=-2`, `D1 = TVD1·exp(η)`),
four doses q12h, sampled 1, 2, 2.5, 3, 6 and 11.5 h after each: with `D1` near 2 h and its IIV
at 40 % CV, the infusion ends of most subjects fall on both sides of the 1-3 h samples, so a
fit moves them across observations throughout its EBE searches. Simulated from the closed form
(superposed zero-order inputs into `dA/dt = -CL/V·A`), outside both engines. Standard library
only; seeded. Run: `python3 moving_infusion_end_gen.py > moving_infusion_end.csv`.
"""
import math
import random

random.seed(3010)
N = 60
CL, V, D1 = 4.0, 30.0, 2.0
OM_CL, OM_V, OM_D1 = 0.09, 0.05, 0.16
PROP_SD = 0.1
AMT, II, NDOSE = 100.0, 12.0, 4
OFFSETS = [1.0, 2.0, 2.5, 3.0, 6.0, 11.5]


def conc(t, cl, v, d1):
    k = cl / v
    c = 0.0
    for j in range(NDOSE):
        tau = t - j * II
        if tau <= 0.0:
            continue
        r = AMT / d1
        if tau <= d1:
            c += r / cl * (1.0 - math.exp(-k * tau))
        else:
            c += r / cl * (1.0 - math.exp(-k * d1)) * math.exp(-k * (tau - d1))
    return c


print("ID,TIME,AMT,RATE,DV,MDV,EVID,CMT")
for i in range(1, N + 1):
    cl = CL * math.exp(random.gauss(0.0, math.sqrt(OM_CL)))
    v = V * math.exp(random.gauss(0.0, math.sqrt(OM_V)))
    d1 = D1 * math.exp(random.gauss(0.0, math.sqrt(OM_D1)))
    for j in range(NDOSE):
        t0 = j * II
        print(f"{i},{t0:g},{AMT:g},-2,0,1,1,1")
        for o in OFFSETS:
            t = t0 + o
            dv = conc(t, cl, v, d1) * (1.0 + random.gauss(0.0, PROP_SD))
            print(f"{i},{t:g},0,0,{dv:.6f},0,0,1")
