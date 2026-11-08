"""Drive tools/mbma_canakinumab_fetch.py against a local fake Europe PMC endpoint.

Used by tests/mbma_canakinumab_fetch_script.rs; never touches the network.

    python3 -I fetch_fake_server.py <fetch script> <out dir> <scenario> [script args...]

Starts an `http.server` on 127.0.0.1 that answers the Nth request with the Nth
entry of the scenario's reply list (the last entry repeats), loads the fetch
script as a module, points its URL at the server and its pins at a synthetic
`data_9studies.csv` (the real data may not be committed, #1913), and runs its
`main()`. Prints one `KEY: value` line per fact for the Rust test to assert on.
"""

import hashlib
import importlib.util
import io
import os
import sys
import threading
import time
import types
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

script, out_dir, scenario, extra = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
WATCHDOG_SECONDS = 25

spec = importlib.util.spec_from_file_location("fetch", script)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def synthetic_csv():
    """A CSV that passes `derive()`: 9 studies, 15 arms, 140 rows."""
    arms = [(s, 0, "placebo") for s in range(1, 10)]
    arms += [(s, 1, m.DRUGS[(s - 1) % 3]) for s in range(1, 7)]
    out = io.StringIO()
    out.write(",".join(m.EXPECTED_COLUMNS) + "\n")
    for k, (study, arm, drug) in enumerate(arms):
        for t in range(10 if k < 5 else 9):
            narm = 50 + k
            out.write(
                f"{study},{arm},{drug},{arm},{t},{10 + t},pub{study},{narm},"
                f"trt,1,2,3,{0.1 * (t + 1) - 0.05 * k}\n"
            )
    return out.getvalue().encode()


def payload(csv_bytes):
    """The endpoint's shape: an outer zip holding the inner zip holding the CSV."""
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr(m.INNER_CSV, csv_bytes)
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("other.pdf", b"%PDF")
        z.writestr(m.INNER_ZIP, inner.getvalue())
    return outer.getvalue()


GOOD = synthetic_csv()
WRONG_A = GOOD + b"\n"
WRONG_B = GOOD + b"\n\n"
TRICKLE = "trickle"
HTTP_500 = "500"
SILENT = "silent"

SCENARIOS = {
    "wrong_then_right": [payload(WRONG_A), payload(GOOD)],
    "garbled_then_right": [b"<html>gateway hiccup</html>", payload(GOOD)],
    "same_wrong": [payload(WRONG_A)],
    "alternating_wrong": [payload(WRONG_A), payload(WRONG_B), payload(WRONG_A)],
    "error_then_same_wrong": [HTTP_500, payload(WRONG_A)],
    "always_500": [HTTP_500],
    "trickle": [TRICKLE],
    "silent": [SILENT],
}
replies = SCENARIOS[scenario]
hits = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        n = len(hits)
        hits.append(n)
        reply = replies[min(n, len(replies) - 1)]
        if reply == HTTP_500:
            self.send_error(500)
            return
        if reply == SILENT:
            # Accept the request and never answer: only a socket timeout ends it.
            time.sleep(WATCHDOG_SECONDS + 5)
            return
        if reply == TRICKLE:
            self.send_response(200)
            self.send_header("Content-Length", "10000000")
            self.end_headers()
            try:
                while True:
                    self.wfile.write(b"P")
                    self.wfile.flush()
                    time.sleep(0.1)
            except OSError:
                return
        self.send_response(200)
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)


def watchdog():
    """A fetch that ignores its deadline would hang the test binary; fail it instead."""
    time.sleep(WATCHDOG_SECONDS)
    print(f"WATCHDOG: the fetch was still running after {WATCHDOG_SECONDS} s", flush=True)
    os._exit(3)


threading.Thread(target=watchdog, daemon=True).start()
server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
server.daemon_threads = True
threading.Thread(target=server.serve_forever, daemon=True).start()

sleeps = []
m.time = types.SimpleNamespace(monotonic=time.monotonic, sleep=lambda s: sleeps.append(s))
m.URL = f"http://127.0.0.1:{server.server_address[1]}/supplementaryFiles"
m.UPSTREAM_SHA256 = hashlib.sha256(GOOD).hexdigest()
m.DERIVED_SHA256 = hashlib.sha256(m.derive(GOOD)).hexdigest()
m.BACKOFF_SECONDS = 1

out = f"{out_dir}/mbma_canakinumab.csv"
sys.argv = ["mbma_canakinumab_fetch.py", "--out", out] + extra
print(f"PIN: {m.UPSTREAM_SHA256}")
print(f"WRONG_A: {hashlib.sha256(WRONG_A).hexdigest()}")
print(f"WRONG_B: {hashlib.sha256(WRONG_B).hexdigest()}")
sys.stdout.flush()
t0 = time.monotonic()
code, message = 0, ""
try:
    m.main()
except SystemExit as e:
    code, message = (1 if isinstance(e.code, str) else e.code or 0), str(e.code)
print(f"ELAPSED: {time.monotonic() - t0:.2f}")
print(f"HITS: {len(hits)}")
print(f"SLEEPS: {len(sleeps)}")
try:
    written = hashlib.sha256(open(out, "rb").read()).hexdigest() == m.DERIVED_SHA256
except OSError:
    written = False
print(f"WROTE_PINNED: {written}")
print(f"EXIT: {code}")
print("MESSAGE-BEGIN")
print(message)
print("MESSAGE-END")
