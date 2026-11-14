"""Drive tools/mbma_canakinumab_fetch.py against a local fake Europe PMC endpoint.

Used by tests/mbma_canakinumab_fetch_script.rs; never touches the network.

    python3 -I fetch_fake_server.py <fetch script> <out dir> <scenario> \
        [--harness-production-timing] [--harness-reserve=F] [script args...]

Starts an `http.server` on 127.0.0.1 that answers the Nth request with the Nth
entry of the scenario's reply list (the last entry repeats), loads the fetch
script as a module, points its URL at the server and its pins at a synthetic
`data_9studies.csv` (the real data may not be committed, #1913), and runs its
`main()`. Prints one `KEY: value` line per fact for the Rust test to assert on.

Timing. By default the backoff is scaled down to waits of 1 s, 2 s, 4 s, ... with
no cap and no reserve, the numbers every test before #2824 was written against.
`--harness-production-timing` keeps the script's own base, cap and reserve (the
virtual clock below makes a 720 s budget cost no wall time); `--harness-reserve=F`
switches the reserve on in the scaled timing. `errorbean_until_<S>` answers with
the errorBean while the virtual clock reads less than S seconds since the fetch
began, then with the right payload.
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
production_timing = "--harness-production-timing" in extra
reserve_opt = [a.split("=", 1)[1] for a in extra if a.startswith("--harness-reserve=")]
unknown = [
    a for a in extra
    if a.startswith("--harness-")
    and a != "--harness-production-timing"
    and not a.startswith("--harness-reserve=")
]
if unknown:
    # A typo here would silently run a different schedule than the test's name says.
    sys.exit(f"fetch_fake_server: unknown harness option(s): {' '.join(unknown)}")
extra = [a for a in extra if not a.startswith("--harness-")]
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
TRICKLE_THEN_STALL = "trickle_then_stall"
TRUNCATED = "truncated"
SLOW_HEADERS = "slow_headers"
HEADERS_FOREVER = "headers_forever"
ERRORBEAN = "errorbean"
LONG_HTML = "long_html"

# A stand-in for the 292-byte reply Europe PMC served on red nights (#2824): its
# shape is the errorBean the endpoint returns for a missing article, sent as
# `application/xml` with HTTP 200; its last 60 bytes are distinctive, so a log
# that quotes only the start of it is caught.
ERRORBEAN_BODY = (
    b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    b'<ns4:errorBean xmlns:ns4="http://www.europepmc.org/data">'
    b"<errCode>500</errCode><errMsg>"
)
ERRORBEAN_TAIL = b"fake errMsg tail for the fetch test</errMsg></ns4:errorBean>"
ERRORBEAN_BODY += b"x" * (292 - len(ERRORBEAN_BODY) - len(ERRORBEAN_TAIL)) + ERRORBEAN_TAIL
assert len(ERRORBEAN_BODY) == 292 and len(ERRORBEAN_TAIL) == 60

# A 2048-byte non-zip reply with one marker inside the first 1024 bytes and one
# after them.
LONG_HTML_BODY = b"<html>" + b"a" * 994 + b"EARLY-MARKER"
LONG_HTML_BODY += b"b" * (1500 - len(LONG_HTML_BODY)) + b"LATE-MARKER"
LONG_HTML_BODY += b"c" * (2048 - len(LONG_HTML_BODY) - 7) + b"</html>"
assert len(LONG_HTML_BODY) == 2048 and LONG_HTML_BODY.index(b"EARLY-MARKER") + 12 <= 1024

SCENARIOS = {
    "wrong_then_right": [payload(WRONG_A), payload(GOOD)],
    "garbled_then_right": [b"<html>gateway hiccup</html>", payload(GOOD)],
    "same_wrong": [payload(WRONG_A)],
    "alternating_wrong": [payload(WRONG_A), payload(WRONG_B), payload(WRONG_A)],
    "error_then_same_wrong": [HTTP_500, payload(WRONG_A)],
    "always_500": [HTTP_500],
    "trickle": [TRICKLE],
    "silent": [SILENT],
    "same_wrong_then_trickle": [payload(WRONG_A), payload(WRONG_A), TRICKLE],
    "trickle_then_stall": [TRICKLE_THEN_STALL],
    "truncated_then_right": [TRUNCATED, payload(GOOD)],
    "slow_headers": [SLOW_HEADERS],
    "headers_forever": [HEADERS_FOREVER],
    "errorbean_then_right": [ERRORBEAN, payload(GOOD)],
    "long_html_then_right": [LONG_HTML, payload(GOOD)],
}
if scenario.startswith("errorbean_until_"):
    until_s = float(scenario.rsplit("_", 1)[1])
    replies = None
else:
    replies = SCENARIOS[scenario]
hits = []
hit_times = []  # virtual seconds since the fetch began, one per request


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        n = len(hits)
        hits.append(n)
        t = virtual_now() - start
        hit_times.append(t)
        if replies is None:
            reply = ERRORBEAN if t < until_s else payload(GOOD)
        else:
            reply = replies[min(n, len(replies) - 1)]
        if reply in (ERRORBEAN, LONG_HTML):
            body, ctype = (
                (ERRORBEAN_BODY, "application/xml")
                if reply == ERRORBEAN
                else (LONG_HTML_BODY, "text/html")
            )
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if reply == HEADERS_FOREVER:
            # One header byte every 0.1 s, never ending: each byte resets a socket
            # timeout, so only the deadline can end the attempt.
            try:
                self.wfile.write(b"HTTP/1.0 200 OK\r\nX-Forever: ")
                while True:
                    self.wfile.write(b"z")
                    self.wfile.flush()
                    time.sleep(0.1)
            except OSError:
                pass
            return
        if reply == HTTP_500:
            self.send_error(500)
            return
        if reply == SILENT:
            # Accept the request and never answer: only a socket timeout ends it.
            time.sleep(WATCHDOG_SECONDS + 5)
            return
        if reply == SLOW_HEADERS:
            # Trickle the response *headers* for ~2.5 s, then send a whole valid body:
            # `urlopen` itself returns after a 2 s deadline, before any body read.
            whole = payload(GOOD)
            try:
                self.wfile.write(b"HTTP/1.0 200 OK\r\nX-Slow: ")
                for _ in range(25):
                    self.wfile.write(b"z")
                    self.wfile.flush()
                    time.sleep(0.1)
                self.wfile.write(f"\r\nContent-Length: {len(whole)}\r\n\r\n".encode())
                self.wfile.write(whole)
            except OSError:
                pass
            self.close_connection = True
            return
        if reply == TRUNCATED:
            # Promise the whole payload, send half of it, close the connection.
            whole = payload(GOOD)
            self.send_response(200)
            self.send_header("Content-Length", str(len(whole)))
            self.end_headers()
            self.wfile.write(whole[: len(whole) // 2])
            self.close_connection = True
            return
        if reply in (TRICKLE, TRICKLE_THEN_STALL):
            self.send_response(200)
            self.send_header("Content-Length", "10000000")
            self.end_headers()
            # TRICKLE_THEN_STALL sends bytes for 1.8 s, then goes silent: a socket
            # timeout set once at open would let the stalled read outlast the deadline.
            until = time.monotonic() + (1.8 if reply == TRICKLE_THEN_STALL else 1e9)
            try:
                while time.monotonic() < until:
                    self.wfile.write(b"P")
                    self.wfile.flush()
                    time.sleep(0.1)
                time.sleep(WATCHDOG_SECONDS + 5)
            except OSError:
                pass
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

# A virtual clock for the script: `sleep` returns at once but moves its `monotonic`
# forward by the requested amount, so the backoff spends deadline without spending
# wall time, and the server threads keep the real `time`.
# The server threads read the same clock, to time-gate replies and log `HIT_TIMES`.
sleeps = []
start = 0.0


def virtual_sleep(s):
    sleeps.append(s)


def virtual_now():
    return time.monotonic() + sum(sleeps)


m.time = types.SimpleNamespace(monotonic=virtual_now, sleep=virtual_sleep)
m.URL = f"http://127.0.0.1:{server.server_address[1]}/supplementaryFiles"
m.UPSTREAM_SHA256 = hashlib.sha256(GOOD).hexdigest()
m.DERIVED_SHA256 = hashlib.sha256(m.derive(GOOD)).hexdigest()
if not production_timing:
    m.BACKOFF_BASE_SECONDS = 1
    m.BACKOFF_CAP_SECONDS = 1e9
    m.RESERVE_FRACTION = float(reserve_opt[0]) if reserve_opt else 0.0

out = f"{out_dir}/mbma_canakinumab.csv"
sys.argv = ["mbma_canakinumab_fetch.py", "--out", out] + extra
print(f"PIN: {m.UPSTREAM_SHA256}")
print(f"WRONG_A: {hashlib.sha256(WRONG_A).hexdigest()}")
print(f"WRONG_B: {hashlib.sha256(WRONG_B).hexdigest()}")
print(f"ERRORBEAN_TAIL: {ERRORBEAN_TAIL.decode()}")
sys.stdout.flush()
t0 = time.monotonic()
start = virtual_now()
code, message = 0, ""
try:
    m.main()
except SystemExit as e:
    code, message = (1 if isinstance(e.code, str) else e.code or 0), str(e.code)
print(f"ELAPSED: {time.monotonic() - t0:.2f}")
print(f"HITS: {len(hits)}")
print(f"HIT_TIMES: {','.join(f'{t:.3f}' for t in hit_times)}")
print(f"SLEEPS: {len(sleeps)}")
print(f"SLEEP_TOTAL: {sum(sleeps):.3f}")
try:
    written = hashlib.sha256(open(out, "rb").read()).hexdigest() == m.DERIVED_SHA256
except OSError:
    written = False
print(f"WROTE_PINNED: {written}")
print(f"EXIT: {code}")
print("MESSAGE-BEGIN")
print(message)
print("MESSAGE-END")
