"""Build the #1876 follow-up fixtures (an occasion only a pk-only row carries) from the
committed X / A ones: the X3, XM3 and AT datasets (NONMEM and ferx spellings) and the
`iov_pk_only_{foce,focei}_{X3,XM3,AT,X3_k0}.ctl` streams. Run from the repo root; writes
into nonmem_anchor/. Read by tests/iov_pk_only_nonmem_anchor.rs."""
import pathlib
import re

D = pathlib.Path("nonmem_anchor")


def rows(path):
    lines = path.read_text().splitlines()
    return lines[0], [line.split(",") for line in lines[1:] if line.strip()]


def write(path, header, rs):
    path.write_text(header + "\n" + "\n".join(",".join(r) for r in rs) + "\n")
    print("wrote", path)


COLS = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,OCC".split(",")
COL = {c: k for k, c in enumerate(COLS)}

for side in ["nm", "ferx"]:
    header, x = rows(D / f"iov_pk_only_{side}_X.csv")
    assert header.split(",") == COLS, header
    # X3: the TIME 110 pk-only row carries OCC 3, which no other record has.
    x3 = []
    n = 0
    for r in x:
        r = list(r)
        if r[COL["EVID"]] == "2":
            assert r[COL["TIME"]] == "110.0" and r[COL["OCC"]] == "2", r
            r[COL["OCC"]] = "3"
            n += 1
        assert r[COL["OCC"]] in ("1", "2", "3")
        x3.append(r)
    write(D / f"iov_pk_only_{side}_X3.csv", header, x3)
    # XM3: the same row written as EVID=0, MDV=1.
    xm3 = []
    for r in x3:
        r = list(r)
        if r[COL["EVID"]] == "2":
            r[COL["EVID"]] = "0"
            assert r[COL["MDV"]] == "1"
        xm3.append(r)
    write(D / f"iov_pk_only_{side}_XM3.csv", header, xm3)

    # AT: A plus one EVID=2 row per subject, 24 h after its last record, under OCC 3.
    # It governs no observation, so κ₃ is uninformed: the degenerate oracle is A's OBJ.
    header_a, a = rows(D / f"iov_pk_only_{side}_A.csv")
    assert header_a.split(",") == COLS, header_a
    pk_only_cmt = x[[r[COL["EVID"]] for r in x].index("2")][COL["CMT"]]
    at = []
    for k, r in enumerate(a):
        at.append(r)
        last = k + 1 == len(a) or a[k + 1][COL["ID"]] != r[COL["ID"]]
        if last:
            t = float(r[COL["TIME"]]) + 24.0
            at.append(
                [r[COL["ID"]], f"{t:.1f}", ".", "2", ".", pk_only_cmt, "0", "1", "3"]
            )
    write(D / f"iov_pk_only_{side}_AT.csv", header, at)
    print(side, "rows changed in X3:", n)

for method in ["foce", "focei"]:
    x_ctl = (D / f"iov_pk_only_{method}_X.ctl").read_text()
    assert "iov_pk_only_nm_X.csv" in x_ctl
    # Three κ: OCC 3 gets ETA(6) of its own, as NONMEM gives any occasion.
    three = x_ctl.replace(
        "  IF(OCC.EQ.2) KAPPA = ETA(5)\n",
        "  IF(OCC.EQ.2) KAPPA = ETA(5)\n  IF(OCC.EQ.3) KAPPA = ETA(6)\n",
    ).replace(
        "$OMEGA BLOCK(1) SAME  ; occ 2\n",
        "$OMEGA BLOCK(1) SAME  ; occ 2\n$OMEGA BLOCK(1) SAME  ; occ 3\n",
    ).replace("ETA4 ETA5 NOPRINT", "ETA4 ETA5 ETA6 NOPRINT")
    assert three.count("ETA(6)") == 1 and "occ 3" in three and "ETA6" in three
    for data in ["X3", "XM3", "AT"]:
        ctl = three.replace("iov_pk_only_nm_X.csv", f"iov_pk_only_nm_{data}.csv")
        ctl = re.sub(r"data=X ", f"data={data} ", ctl)
        (D / f"iov_pk_only_{method}_{data}.ctl").write_text(ctl)
        print("wrote", f"iov_pk_only_{method}_{data}.ctl")
    # X3_k0: X's two-κ stream on X3 data, so OCC 3 falls to KAPPA = 0 — ferx before
    # this PR.
    ctl = x_ctl.replace("iov_pk_only_nm_X.csv", "iov_pk_only_nm_X3.csv")
    ctl = ctl.replace("data=X ", "data=X3 ")
    ctl = ctl.replace("variant=own-OCC", "variant=k0-unslotted")
    (D / f"iov_pk_only_{method}_X3_k0.ctl").write_text(ctl)
    print("wrote", f"iov_pk_only_{method}_X3_k0.ctl")
