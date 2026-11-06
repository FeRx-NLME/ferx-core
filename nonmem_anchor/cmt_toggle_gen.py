#!/usr/bin/env python3
"""Regenerate cmt_toggle.csv: one subject per measured compartment on/off case (#2732).

Run from nonmem_anchor/. `tests/cmt_toggle_nonmem_anchor.rs` names each ID.

Two passes. With no argument the DVs are placeholders; run NONMEM, then pass its table
(`cmt_toggle_gen.py results/cmt_toggle.tab`) to set each DV to PRED plus a fixed offset
that alternates in sign, and run NONMEM again. That keeps the EBEs near zero and the
objective sensitive to a wrong prediction.
"""
import sys

HDR = "ID,TIME,DV,EVID,AMT,CMT,MDV,RATE,II,SS,ADDL,KBC,LAGC,WT,A0C\n"


def row(t, dv=0, evid=0, amt=0, cmt=0, mdv=None, rate=0, ii=0, addl=0, wt=70):
    if mdv is None:
        mdv = 0 if evid == 0 else 1
    return dict(t=t, dv=dv, evid=evid, amt=amt, cmt=cmt, mdv=mdv, rate=rate, ii=ii, addl=addl, wt=wt)


def probe(t, wt=70):
    return row(t, evid=2, cmt=0, wt=wt)


def on(t, c=2, wt=70):
    return row(t, evid=2, cmt=c, wt=wt)


def off(t, c=2, wt=70):
    return row(t, evid=2, cmt=-c, wt=wt)


def dose(t, a=100, c=1, **k):
    return row(t, evid=1, amt=a, cmt=c, **k)


def obs(t, c=1, dv=50, wt=70):
    return row(t, dv=dv, cmt=c, wt=wt)


def reset(t, c=1):
    return row(t, evid=3, cmt=c)


# (rows, KB, ALAG2, A_0(2))
CASES = [
    # 1 the issue's reproduction
    ([dose(0), obs(2, 2, 17), obs(4, 2, 30), off(4), on(6), obs(8, 2, 10)], 0, 0, 0),
    # 2 several windows, residual drug at every toggle, a second dose in between
    ([dose(0), obs(2, 2, 17), obs(4, 2, 30), off(4), on(4.5), obs(6, 2, 10), off(6), on(7),
      dose(7.5), obs(9, 2, 20), off(9), on(10), obs(12, 2, 20)], 0, 0, 0),
    # 3 back-transfer out of an off compartment reads zero, and inflow is lost
    ([dose(0), obs(2), obs(4), off(4), obs(5), obs(6), on(7), obs(7), obs(8, 2, 10), obs(10)], 0.3, 0, 0),
    # 4 a bolus into an off compartment turns it on
    ([dose(0), obs(2, 2, 20), off(3), dose(5, 50, 2), obs(6, 2, 50), obs(8, 2, 50)], 0, 0, 0),
    # 5 off at the subject's first record, ahead of a co-timed dose
    ([off(0), dose(0), probe(2), on(3), obs(6, 2, 10)], 0, 0, 0),
    # 6 an infusion into the compartment is cancelled, and does not resume
    ([dose(0, 100, 2, rate=10), obs(2, 2, 20), off(3), on(5), obs(6, 2, 10), obs(9, 2, 10), obs(14, 2, 10)], 0, 0, 0),
    # 7 the same into central, with the rest of the system still running
    ([dose(0, 100, 1, rate=10), obs(2), off(3, 1), on(5, 1), obs(6), obs(9), obs(14)], 0, 0, 0),
    # 8 an infusion into an off compartment turns it on at its start
    ([dose(0), off(1), dose(2, 100, 2, rate=10), obs(3, 2, 10), obs(6, 2, 10), obs(14, 2, 10)], 0, 0, 0),
    # 9 an infusion upstream of the off compartment carries on
    ([dose(0, 100, 1, rate=10), off(1), probe(2), probe(4), on(5), obs(6, 2, 10), obs(12)], 0, 0, 0),
    # 10 a lagged dose in flight at the off row never lands
    ([dose(0), obs(2, 2, 20), dose(3, 50, 2), off(3.5), probe(4.5), on(5), obs(6, 2, 10)], 0, 1, 0),
    # 11 a lagged dose into an off compartment turns it on at the arrival, not the record
    ([dose(0), off(2), dose(3, 50, 2), probe(3.5), probe(4.5), obs(6, 2, 10)], 0, 1, 0),
    # 12 an observation between that record and arrival is accepted and reads zero
    ([dose(0), off(2), dose(3, 50, 2), obs(3.5, 2, 0), obs(4.5, 2, 10)], 0, 1, 0),
    # 13 a second off row changes nothing
    ([dose(0), obs(2, 2, 17), off(3), off(4), on(5), obs(8, 2, 10)], 0, 0, 0),
    # 14 an on row on a compartment that is on changes nothing
    ([dose(0), obs(2, 2, 17), on(4), obs(8, 2, 50)], 0, 0, 0),
    # 15 a reset turns the compartment on
    ([dose(0), obs(2, 2, 17), off(3), reset(4), dose(4.5), obs(6, 2, 10)], 0, 0, 0),
    # 16 EVID=4 turns it on too
    ([dose(0), off(2), row(3, evid=4, amt=100, cmt=1), obs(5, 2, 10)], 0, 0, 0),
    # 17 dose, then off, at one TIME: the dose is wiped
    ([dose(0), dose(4, 20, 2), off(4), probe(5), on(6), obs(7, 2, 10)], 0, 0, 0),
    # 18 observe, off, on, observe at one TIME
    ([dose(0), obs(4, 2, 30), off(4), on(4), obs(4, 2, 0), obs(6, 2, 10)], 0, 0, 0),
    # 19 off, then a dose into another compartment, at one TIME
    ([dose(0), off(4), dose(4, 20, 1), probe(5), on(6), obs(7, 2, 10)], 0, 0, 0),
    # 20 off, then a dose into the same compartment, at one TIME
    ([dose(0), obs(2), off(4, 1), dose(4, 20, 1), obs(6)], 0, 0, 0),
    # 21 the rest of an ADDL train into the compartment never arrives
    ([dose(0, 100, 1), dose(0, 10, 2, ii=2, addl=3), obs(1, 2, 10), off(3), on(5), obs(9, 2, 10)], 0, 0, 0),
    # 22 an ADDL train into another compartment is unaffected
    ([dose(0, 10, 1, ii=2, addl=3), off(3), probe(4.5), on(7), obs(8, 2, 1), obs(8)], 0, 0, 0),
    # 23 the off row is a record: its covariates govern the interval ending at it
    ([dose(0), obs(2, 2, 20), off(4, wt=140), on(6), obs(8, 2, 10, wt=35), obs(10, 2, 10, wt=35)], 0, 0, 0),
    # 24 on restarts from zero and does not re-apply the initial amount
    ([dose(0), obs(2, 2, 17), off(3), on(4), obs(6, 2, 10)], 0, 0, 5),
]


def preds(path):
    """PRED of each observation record, in file order."""
    out = []
    with open(path) as f:
        for line in f.readlines()[2:]:
            c = line.split()
            if float(c[2]) == 0:
                out.append(float(c[7]))
    return out


def main():
    pred = iter(preds(sys.argv[1])) if len(sys.argv) > 1 else None
    n_obs = 0
    lines = [HDR]
    for i, (rows, kb, lag, a0) in enumerate(CASES, 1):
        for r in rows:
            if pred is not None and r["evid"] == 0:
                sign = 1 if n_obs % 2 == 0 else -1
                n_obs += 1
                r["dv"] = round(next(pred) * (1 + 0.05 * sign) + 0.2 * sign, 6)
            lines.append(
                f"{i},{r['t']},{r['dv']},{r['evid']},{r['amt']},{r['cmt']},{r['mdv']},{r['rate']},"
                f"{r['ii']},0,{r['addl']},{kb},{lag},{r['wt']},{a0}\n"
            )
    with open("cmt_toggle.csv", "w") as f:
        f.writelines(lines)


main()
