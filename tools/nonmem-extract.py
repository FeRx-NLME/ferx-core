#!/usr/bin/env python3
"""Extract the numbers of a NONMEM listing into JSON (#3182).

    tools/nonmem-extract.py run.lst [more.lst ...] [--out-dir DIR]

Writes `<stem>.nm.json` next to each listing, or in DIR. Commit that, the control
stream, the data and the numeric tables (.ext, .phi, .tab, .cov); never the listing,
which carries NONMEM's own text and the license holder's details.

The result holds our control stream as NONMEM echoed it, record counts, warning
numbers, failure markers, and per estimation step: the objective, termination
markers, shrinkage, final estimates, standard errors and eigenvalues, at the
precision NONMEM printed. `listing_truncated` / a step's `report_truncated` mark a
listing that ends before the estimation output / that step's report.
"""

import argparse
import json
import pathlib
import re
import sys

NUM = re.compile(r"[-+]?(?:\d+\.?\d*(?:[EeDd][-+]?\d+)?|\.\d+(?:[EeDd][-+]?\d+)?|INF|NaN)$")
MARKERS = [
    "PRDERR",
    "NUMERICAL DIFFICULTIES",
    "PROGRAM TERMINATED",
    "MINIMIZATION SUCCESSFUL",
    "MINIMIZATION TERMINATED",
    "PROBLEMS OCCURRED WITH THE MINIMIZATION",
    "ROUNDING ERRORS",
]
SECTIONS = {
    "FINAL PARAMETER ESTIMATE": "final",
    "STANDARD ERROR OF ESTIMATE": "standard_error",
    "COVARIANCE MATRIX OF ESTIMATE": "covariance",
    "EIGENVALUES OF COR MATRIX OF ESTIMATE": "eigenvalues",
}
STATS = re.compile(
    r"^ (ETABAR|SE|P VAL\.|ETASHRINKSD|ETASHRINKVR|EBVSHRINKSD|EBVSHRINKVR|"
    r"RELATIVEINF|EPSSHRINKSD|EPSSHRINKVR)(?::|\(%\)) *(.*)$"
)


def number(tok):
    """A JSON number; a non-finite value stays a string, a dotted cell is None."""
    if set(tok) == {"."}:
        return None
    if re.search(r"INF|NaN", tok):
        return tok
    value = float(tok.replace("D", "E").replace("d", "e"))
    return int(tok) if re.fullmatch(r"[-+]?\d+", tok) else value


def numbers(text):
    """The numbers on a line, or None if it holds anything else."""
    toks = text.split()
    ok = toks and all(NUM.match(t) or set(t) == {"."} for t in toks)
    return [number(t) for t in toks] if ok else None


def after(text, label):
    m = re.search(re.escape(label) + r"\s*(\S+)", text)
    return number(m.group(1)) if m and NUM.match(m.group(1)) else None


def rows(lines):
    """`{row label: values}` of a triangular block, wrapped rows joined."""
    out, label, open_row = {}, None, False
    for line in lines:
        if line.startswith("+"):
            out.setdefault(label, []).extend(numbers(line[1:]) or [])
            open_row = True
        elif not line.strip():
            continue
        elif open_row and numbers(line) is not None:
            out[label].extend(numbers(line))
        else:
            label, open_row = line.strip(), False
    return out


def estimates(lines):
    """THETA / OMEGA / SIGMA of a final-estimate or standard-error section."""
    out, name, block = {}, None, []

    def close():
        if name == "theta":
            values = [v for line in block for v in numbers(line) or []]
            out["theta"] = values
        elif name:
            out[name] = rows(block)

    for line in lines:
        m = re.match(r" (THETA|OMEGA|SIGMA) - (VECTOR|COV|CORR)", line)
        if m:
            close()
            name = None if m.group(2) == "CORR" else m.group(1).lower()
            block = []
        else:
            block.append(line)
    close()
    return out


def sections(text):
    """`(kind, lines)` of every starred section of an estimation step."""
    found, kind, body = [], None, []
    for line in text.splitlines():
        # A rule, or the form-feed "1" that starts a printed page.
        if re.match(r" \*{100}|1\s*$", line):
            continue
        m = re.match(r" \*{20} +(.*?) +\*{20}$", line)
        if m and not m.group(1).strip(" *"):
            continue
        if m:
            title = re.sub(r" \(\w+\)$", "", m.group(1).strip())
            if kind and body:
                found.append((kind, body))
            kind, body = SECTIONS.get(title), []
        elif re.match(r" #\w+:", line):
            if kind and body:
                found.append((kind, body))
            kind, body = None, []
        elif kind:
            body.append(line)
    if kind and body:
        found.append((kind, body))
    return found


def step(text, keep_covariance):
    out = {"method": text.splitlines()[0].strip()}
    m = re.search(r"#OBJV:\**\s*(\S+?)\s*\*", text)
    if m:
        out["objv"] = number(m.group(1))
    for key, label in [
        ("objv_without_constant", "OBJECTIVE FUNCTION VALUE WITHOUT CONSTANT:"),
        ("objv_with_constant", "OBJECTIVE FUNCTION VALUE WITH CONSTANT:"),
        ("function_evaluations", "NO. OF FUNCTION EVALUATIONS USED:"),
        ("significant_digits", "NO. OF SIG. DIGITS IN FINAL EST.:"),
    ]:
        value = after(text, label)
        if value is not None:
            out[key] = value
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = STATS.match(line)
        if not m:
            continue
        key = m.group(1).lower().replace(" ", "_").rstrip(".")
        if key == "se":
            key = "etabar_se"
        values = numbers(m.group(2)) or []
        for more in lines[i + 1 :]:
            if not more.strip() or numbers(more) is None:
                break
            values += numbers(more)
        out.setdefault(key, []).append(values)
    for kind, body in sections(text):
        if kind in ("final", "standard_error"):
            out[kind] = estimates(body)
        elif kind == "eigenvalues":
            data = [numbers(line) for line in body if numbers(line)]
            out[kind] = [v for row in data for v in row if not isinstance(v, int)]
        elif keep_covariance:
            out[kind] = rows(body)
    out["markers"] = [m for m in MARKERS if m in text]
    if "objv" not in out and "final" not in out:
        # The listing ends before this step's report.
        out["report_truncated"] = True
    return out


def extract(text, keep_covariance):
    parts = re.split(r"^ *NM-TRAN MESSAGES|^ *License Registered", text, maxsplit=1, flags=re.M | re.I)
    if len(parts) < 2:
        # Without the boundary the "control stream" would run to the end of the listing
        # and carry NONMEM's text and the license holder into the JSON: refuse instead.
        raise ValueError("cannot find where the control stream echo ends")
    head = parts[0]
    start = re.search(r"^\s*(?:\$|;)", head, re.M)
    control = head[start.start() :].strip("\n") if start else ""
    out = {}
    m = re.search(r"\(NONMEM\) VERSION (\S+)", text)
    if m:
        out["nonmem_version"] = m.group(1)
    m = re.search(r"^\s*\$DATA\s+(\S+)", control, re.M)
    if m:
        out["data_file"] = m.group(1)
    for key, label in [
        ("data_records", "NO. OF DATA RECS IN DATA SET:"),
        ("observations", "TOT. NO. OF OBS RECS:"),
        ("individuals", "TOT. NO. OF INDIVIDUALS:"),
    ]:
        value = after(text, label)
        if value is not None:
            out[key] = value
    out["nmtran_warnings"] = sorted({int(n) for n in re.findall(r"^ \(WARNING +(\d+)\)", text, re.M)})
    output = text[len(head) :]
    out["markers"] = [m for m in MARKERS if m in output]
    out["steps"] = [step(s, keep_covariance) for s in re.split(r"^ #METH:", text, flags=re.M)[1:]]
    if not out["steps"] and re.search(r"^\s*\$EST", control, re.M | re.I):
        # The listing stops before estimation output (e.g. only the NM-TRAN part was kept);
        # the numbers are then in the .ext alone.
        out["listing_truncated"] = True
    out["control_stream_as_run"] = [line.rstrip() for line in control.splitlines()]
    if re.search(r"license|registered to", control, re.I):
        raise ValueError("the control stream echo contains license text")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("listings", nargs="+", type=pathlib.Path)
    ap.add_argument("--out-dir", type=pathlib.Path)
    args = ap.parse_args()
    for lst in args.listings:
        try:
            result = extract(lst.read_text(errors="replace"), not lst.with_suffix(".cov").exists())
        except ValueError as e:
            sys.exit(f"{lst}: {e}")
        out = (args.out_dir or lst.parent) / f"{lst.stem}.nm.json"
        # One line per innermost list, so the diff of a re-run stays readable.
        body = json.dumps(result, indent=1)
        body = re.sub(
            r"\[\s+([^\[\]{}\"]*?)\s+\]",
            lambda m: "[" + re.sub(r",\s+", ", ", m.group(1)) + "]",
            body,
        )
        out.write_text(body + "\n")


if __name__ == "__main__":
    main()
