# Pharmpy variability-search anchors (#1183)

The external references for `ferx iivsearch` and `ferx iovsearch` are
Pharmpy 2.2.0's own runs of `iivsearch` / `iovsearch`, driving NONMEM 7.5.1,
on simulated datasets whose variability structure has something to find.
The inputs, the run scripts and the serialised trajectories live under
`crates/ferx-tools/tests/pharmpy/{iivsearch,iovsearch}_anchor/` (each has a
README with the numbers), and the slow-gated tests
`crates/ferx-tools/tests/{iivsearch,iovsearch}_pharmpy_anchor.rs` replay
ferx on the same inputs.

`pmx.sh` is the one piece of tooling: it runs a command inside the licensed
`pmx` NONMEM container, in one of those anchor directories — the host repo
is mounted at the same path under `/home/rstudio`, and Pharmpy is
pip-installed in the container's user site with
`~/.config/Pharmpy/pharmpy.conf` pointing at `/opt/NONMEM/nm751`.

```bash
# simulate the data (numpy lives in the container, not on the host)
tools/pharmpy-variability-anchor/pmx.sh iivsearch_anchor python3 simulate_iiv.py
tools/pharmpy-variability-anchor/pmx.sh iovsearch_anchor python3 simulate_iov.py

# regenerate the trajectories (NONMEM; a few minutes each)
tools/pharmpy-variability-anchor/pmx.sh iivsearch_anchor python3 run_iivsearch.py
tools/pharmpy-variability-anchor/pmx.sh iovsearch_anchor python3 run_iovsearch.py

# replay ferx against them
cargo test -p ferx-tools --release --features slow-tests \
  --test iivsearch_pharmpy_anchor --test iovsearch_pharmpy_anchor -- --nocapture
```

The container is only needed to *regenerate*; CI replays the committed
JSON. Pharmpy's run directories (`base_fit/`, `iivsearch_*/`,
`iovsearch_*/`) are large and are not committed — delete them after the
JSON is written.
