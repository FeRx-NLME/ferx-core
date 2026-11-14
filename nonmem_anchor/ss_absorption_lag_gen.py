# Generates ss_absorption_lag.csv (`python3 ss_absorption_lag_gen.py > ss_absorption_lag.csv`).
"""Dataset for nonmem_anchor/ss_zero_order_lag.ctl and ss_first_order_lag.ctl.

SS=1 doses (II = 8) into an ABSORPTION input, with and without a lag, each followed by a
non-SS dose at t = 24 landing on residual drug. Columns DR (zero-order duration) and LG
(lag) are per subject; WT steps from 70 to 50 at t = 10 for the stepped subjects. DV is 0:
the anchors compare PRED, which does not read DV.

IDs 1-9 feed the zero-order stream (CMT 1 = central, RATE = -2, D1 = DR):
  1: DR 2,  LG 0           2: DR 6,  LG 0          3: DR 8 (= II), LG 0
  4: DR 12 (> II), LG 0    5: DR 2,  LG 3          6: DR 6, LG 5 (lag + dur > II)
  7: DR 2,  LG 9 (>= II)   8: DR 6,  LG 3, WT step 9: DR 12, LG 8 (= II), WT step
IDs 11-17 feed the first-order stream (CMT 1 = depot, CMT 2 = central, KA):
  11: bolus, LG 0          12: bolus, LG 2         13: bolus, LG 5
  14: bolus, LG 8 (= II)   15: bolus, LG 11 (> II) 16: bolus, LG 3, WT step
  17: infusion RATE 100 into the depot (T_inf 10 > II), LG 3
"""
OBS = [0.5, 1, 2, 3, 4, 5.5, 7, 7.9, 9, 10, 12, 14, 16, 18, 20, 22, 25, 26, 28, 30, 34, 40, 48]
ZO = [(1, 2, 0, False), (2, 6, 0, False), (3, 8, 0, False), (4, 12, 0, False),
      (5, 2, 3, False), (6, 6, 5, False), (7, 2, 9, False), (8, 6, 3, True), (9, 12, 8, True)]
FO = [(11, 0, 0, False), (12, 0, 2, False), (13, 0, 5, False), (14, 0, 8, False),
      (15, 0, 11, False), (16, 0, 3, True), (17, 100, 3, False)]


def subject(i, rate, lag, dur, step, obs_cmt):
    def wt(t):
        return 50.0 if (step and t >= 10) else 70.0

    print(f"{i},0,.,1000,{rate},1,1,1,{wt(0)},1,8,{lag},{dur}")
    rows = [(t, "obs") for t in OBS] + [(24, "dose")]
    for t, k in sorted(rows, key=lambda r: (r[0], r[1] != "dose")):
        if k == "dose":
            print(f"{i},{t},.,400,{rate},1,1,1,{wt(t)},0,0,{lag},{dur}")
        else:
            print(f"{i},{t},0,.,.,0,{obs_cmt},0,{wt(t)},0,0,{lag},{dur}")


print("ID,TIME,DV,AMT,RATE,EVID,CMT,MDV,WT,SS,II,LG,DR")
for i, dur, lag, step in ZO:
    subject(i, -2, lag, dur, step, 1)
for i, rate, lag, step in FO:
    subject(i, rate, lag, 0, step, 2)
