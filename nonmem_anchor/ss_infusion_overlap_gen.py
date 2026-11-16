# Generates ss_infusion_overlap.csv (`python3 ss_infusion_overlap_gen.py > ss_infusion_overlap.csv`).
"""Dataset for nonmem_anchor/ss_infusion_overlap*: SS=1 infusions with T_inf > II."""
OBS = [0.5, 2, 4, 6, 7.9, 10, 12, 14, 18, 20, 22, 26, 30, 34, 40, 48]
# (id, amt, lag, wt step at 10?)  II = 8, RATE = 50
CASES = [(1, 600, 0, False), (2, 1000, 0, False), (3, 800, 0, False),
         (4, 1000, 3, False), (5, 1000, 0, True), (6, 600, 3, True)]
print("ID,TIME,DV,AMT,RATE,EVID,CMT,MDV,WT,SS,II,LG")
for i, amt, lag, step in CASES:
    wt = lambda t: 50.0 if (step and t >= 10) else 70.0
    print(f"{i},0,.,{amt},50,1,1,1,{wt(0)},1,8,{lag}")
    rows = [(t, "obs") for t in OBS] + [(24, "dose")]
    for t, k in sorted(rows, key=lambda r: (r[0], r[1] != "dose")):
        if k == "dose":
            print(f"{i},{t},.,400,50,1,1,1,{wt(t)},0,0,{lag}")
        else:
            print(f"{i},{t},0,.,.,0,1,0,{wt(t)},0,0,{lag}")
