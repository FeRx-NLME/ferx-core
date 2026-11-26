# PsN `scm` anchor

`results.csv` holds the numbers PsN 5.7.1 with NONMEM 7.6 reported on 2026-09-05 for
`base.mod` and `scm.conf`. In the backward step, `selected` would mean removed; neither
relation was. The covariate statistics it used: `WT` median 70.85 (45.0 to 93.7), `CRCL`
median 82.90 (46.5 to 150.0).

Regenerate, then transcribe the numbers; do not commit the log PsN writes:

```bash
scm scm.conf -directory=psn_run -threads=8
```
