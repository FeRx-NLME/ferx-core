"""Compare uninstrumented executables, then collect counts separately.

Usage: python tools/test-focei-profiles.py BEFORE_TIME AFTER_TIME BEFORE_ALLOC AFTER_ALLOC OUTPUT
Executables are built from examples/focei_profile.rs. Run from the repo root.
"""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import csv


def main():
    before_time, after_time, before_alloc, after_alloc, output = map(Path, sys.argv[1:])
    output.mkdir(parents=True, exist_ok=True)
    signatures = {}
    reports = []

    def run(binary, label, case, threads, repeats, mode, name, provider_timers):
        target = output / (name + ".json")
        env = dict(os.environ, FERX_PROFILE=str(int(provider_timers)))
        result = subprocess.run([str(binary.resolve()), case, str(threads), str(repeats), mode, str(target)],
                                env=env, text=True, capture_output=True)
        (output / (name + ".log")).write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{name} failed: {result.stderr}")
        data = json.loads(target.read_text())
        assert data["allocation_build"] == (mode in ("counts", "stacks", "byte_stacks"))
        assert signatures.setdefault(case, data["signature"]) == data["signature"], f"Numerical mismatch: {name}"
        reports.append(dict(label=label, **data))
        return data

    summary = []
    for case in ("analytical", "iov", "stiff"):
        for threads in (1, 4):
            print(f"Timing {case}, {threads} threads (ABBA)", flush=True)
            values = {"before": [], "after": []}
            for i, (label, binary, repeats) in enumerate([
                ("before", before_time, 3), ("after", after_time, 3),
                ("after", after_time, 4), ("before", before_time, 4),
            ]):
                data = run(binary, label, case, threads, repeats, "time",
                           f"timing-{case}-{threads}-{i+1}-{label}", False)
                values[label].extend(row["elapsed_ms"] for row in data["measurements"])
            assert len(values["before"]) == len(values["after"]) == 7
            before, after = (statistics.median(values[label]) for label in ("before", "after"))
            summary.append(dict(case=case, threads=threads, before_ms=before, after_ms=after,
                                speedup=before / after,
                                before_min_ms=min(values["before"]), before_max_ms=max(values["before"]),
                                after_min_ms=min(values["after"]), after_max_ms=max(values["after"])))
    # Instrumentation is never enabled in the timing comparisons above.
    allocation_summary = []
    for case in ("analytical", "iov", "stiff"):
        print(f"Profiling {case}: counters and pointwise kernels", flush=True)
        before = run(before_alloc, "before", case, 1, 1, "counts", f"counts-{case}-before", True)
        after = run(after_alloc, "after", case, 1, 1, "counts", f"counts-{case}-after", True)
        for label, binary in [("before", before_time), ("after", after_time)]:
            run(binary, label, case, 1, 1, "kernels", f"kernels-{case}-{label}", False)
        b = before["measurements"][0]["allocations"]
        a = after["measurements"][0]["allocations"]
        allocation_summary.append(dict(case=case, before_allocs=b["allocation_calls"],
            after_allocs=a["allocation_calls"], before_bytes=b["requested_bytes"], after_bytes=a["requested_bytes"],
            bytes_reduction_percent=100 * (1 - a["requested_bytes"] / b["requested_bytes"])))
    for filename, rows in [("timing-summary.csv", summary), ("allocation-summary.csv", allocation_summary)]:
        with (output / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    metadata = {"signatures_match_across_versions_threads_and_instrumentation": True,
                "timing_repeats_per_version": 7, "order_per_case": "before, after, after, before",
                "hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (before_time, after_time, before_alloc, after_alloc)}}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary + allocation_summary, indent=2))


if __name__ == "__main__":
    main()
