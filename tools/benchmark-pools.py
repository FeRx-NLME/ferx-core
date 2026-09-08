"""Compare prebuilt pool_benchmark executables in ABBA order; stdlib only.

Build examples/pool_benchmark.rs on both revisions with the same Cargo profile.
Usage: python tools/benchmark-pools.py BASELINE_EXE MODIFIED_EXE OUTPUT_DIRECTORY [--reverse]
No compilation runs while timings are collected. Each process has an untimed
reference fit and a reported warmup round (0), excluded from the summary.
"""
import csv
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys


def main():
    baseline, modified, output = map(Path, sys.argv[1:4])
    if sys.argv[4:] not in ([], ["--reverse"]):
        raise SystemExit("optional fourth argument: --reverse")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    signatures = {}
    expected_cases = None
    order = [
        ("baseline", baseline, 3),
        ("modified", modified, 3),
        ("modified", modified, 4),
        ("baseline", baseline, 4),
    ]
    if sys.argv[4:]:
        order = [("modified", modified, 3), ("baseline", baseline, 3),
                 ("baseline", baseline, 4), ("modified", modified, 4)]
    for run, (label, binary, rounds) in enumerate(order):
        print(f"Run {run + 1}/4: {label}, {rounds} measured rounds per case", flush=True)
        completed = subprocess.run([str(binary.resolve()), label, str(rounds)],
                                   capture_output=True, text=True, check=False)
        (output / f"run-{run + 1}.log").write_text(
            completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"{label} failed; see run-{run + 1}.log")
        lines = [line for line in completed.stdout.splitlines()
                 if line.startswith("label,") or line.startswith(label + ",")]
        rows = list(csv.DictReader(lines))
        cases = {row["case"] for row in rows}
        if expected_cases is None:
            expected_cases = cases
        if not cases or cases != expected_cases or len(rows) != len(cases) * (rounds + 1):
            raise RuntimeError(f"Unexpected result count: {len(rows)}")
        with (output / f"run-{run + 1}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for row in rows:
            case = row["case"]
            signature = json.loads(row["signature"])
            if signatures.setdefault(case, signature) != signature:
                raise RuntimeError(f"Numerical mismatch across revisions for {case}")
            if int(row["round"]) > 0:
                records.append(row)
    summaries = []
    for case in signatures:
        values = {label: [float(r["elapsed_ms"]) for r in records
                          if r["case"] == case and r["label"] == label]
                  for label in ("baseline", "modified")}
        assert all(len(v) == 7 for v in values.values())
        before, after = (statistics.median(values[label])
                         for label in ("baseline", "modified"))
        summaries.append({
            "case": case, "baseline_ms": before, "modified_ms": after,
            "speedup": before / after, "time_saved_percent": 100 * (1 - after / before),
            "baseline_min_ms": min(values["baseline"]),
            "baseline_max_ms": max(values["baseline"]),
            "modified_min_ms": min(values["modified"]),
            "modified_max_ms": max(values["modified"]),
        })
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    metadata = {
        "platform": platform.platform(), "python": platform.python_version(),
        "order": ", ".join(label for label, _, _ in order),
        "measured_rounds_per_revision": 7, "all_signatures_bit_identical": True,
        "binary_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (baseline, modified)},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("\nAll objective/parameter/iteration signatures match across revisions.")
    for r in summaries:
        print(f"{r['case']}: {r['baseline_ms']:.2f} -> {r['modified_ms']:.2f} ms; "
              f"{r['speedup']:.2f}x; {r['time_saved_percent']:.1f}% less time")


if __name__ == "__main__":
    main()
