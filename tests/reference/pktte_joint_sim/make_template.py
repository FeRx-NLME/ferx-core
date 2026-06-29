#!/usr/bin/env python3
# Build the shared simulation template for the Slice 2.2 event-time anchor.
# N subjects, single oral dose to CMT 1 at t=0, then a DENSE grid of CMT-3 rows so
# NONMEM integrates the cumulative hazard CHZ(t) on a fine grid (R inverts CHZ=-log U
# per subject for the event time). Same truth/design fed to ferx and rxode2.
N        = 500
DOSE     = 100.0
HORIZON  = 24.0
STEP     = 0.25                      # CHZ output grid (96 obs rows/subject)
rows = ["ID,TIME,DV,EVID,AMT,CMT,MDV"]
g = [round(STEP*k, 4) for k in range(1, int(HORIZON/STEP)+1)]   # 0.25 .. 24.0
for i in range(1, N+1):
    rows.append(f"{i},0,.,1,{DOSE:g},1,1")                      # dose -> depot (CMT 1)
    for t in g:
        rows.append(f"{i},{t:g},0,0,0,3,0")                     # CHZ probe row (CMT 3)
open("simtemplate.csv","w").write("\n".join(rows)+"\n")
print(f"wrote simtemplate.csv: N={N} dose={DOSE} horizon={HORIZON} step={STEP} "
      f"-> {len(rows)-1} data rows ({1+len(g)} per subject)")
