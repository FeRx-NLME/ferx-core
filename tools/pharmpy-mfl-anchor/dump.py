"""Dump Pharmpy's reading of the MFL corpus, and its `expand()` of a set of
symbol-bearing statements on a reference model, as JSON (#1179).

Runs *inside* the Pharmpy container — see run.sh. Two sections:

* ``parse``: for every line of corpus.txt, the raw statement list from
  ``pharmpy.tools.mfl.parse.parse`` (no default-filling, no model), or the
  exception class when Pharmpy rejects it.
* ``expand``: for every statement in EXPAND_CASES, the covariate tuples
  ``ModelFeatures.create_from_mfl_string(s).expand(model)`` produces on a
  basic one-peripheral oral model with WT / CRCL continuous and SEX
  categorical — the oracle for ferx's @-symbol resolution and Pharmpy's
  explicit-beats-symbol / forced-beats-optional rules.

The output is committed at crates/ferx-tools/tests/data/mfl_pharmpy_anchor.json
and read by crates/ferx-tools/src/search/mfl_pharmpy_anchor.rs.
"""

import dataclasses
import json
import sys

import pandas as pd
import pharmpy
from pharmpy.model import DataVariable
from pharmpy.modeling import (
    create_basic_pk_model,
    get_individual_parameters,
    get_pk_parameters,
    set_covariates,
    set_peripheral_compartments,
)
from pharmpy.tools.mfl.parse import ModelFeatures, parse
from pharmpy.tools.mfl.statement.feature.symbols import Name, Option, Wildcard

EXPAND_CASES = [
    "COVARIATE?(@IIV,@CONTINUOUS,[pow,lin])",
    "COVARIATE?(@IIV,@CATEGORICAL,cat)",
    "COVARIATE?(@PK,@CONTINUOUS,*)",
    "COVARIATE?(@PK_IIV,WT,pow)",
    "COVARIATE?(@ABSORPTION,WT,pow);COVARIATE?(@DISTRIBUTION,WT,cat)",
    "COVARIATE?(@ELIMINATION,CRCL,exp)",
    "COVARIATE?(@PD,WT,pow)",
    "LET(CONTINUOUS,[WT]);COVARIATE?(@IIV,@CONTINUOUS,pow)",
    "LET(MINE,[CL,VP1]);COVARIATE?(@MINE,WT,exp)",
    "COVARIATE?(CL,WT,[pow,lin]);COVARIATE(CL,WT,pow)",
    "COVARIATE(@IIV,WT,pow);COVARIATE(@ELIMINATION,WT,exp)",
    "COVARIATE?(@PK,@CONTINUOUS,[pow,lin]);COVARIATE(CL,WT,exp)",
    "COVARIATE?(@IIV,[WT,CRCL],*);COVARIATE?(CL,WT,exp)",
    "COVARIATE?(@NOSUCH,WT,pow)",
    "COVARIATE(CL,WT,[pow,exp])",
]


def convert(value):
    if isinstance(value, Wildcard):
        return "*"
    if isinstance(value, Option):
        return value.option
    if isinstance(value, Name):
        return value.name
    if type(value).__name__ == "Ref":
        return "@" + value.name
    if isinstance(value, (tuple, list)):
        return [convert(v) for v in value]
    if dataclasses.is_dataclass(value):
        out = {"type": type(value).__name__.upper()}
        for f in dataclasses.fields(value):
            out[f.name] = convert(getattr(value, f.name))
        return out
    return value


def dump_parse(corpus_path):
    entries = []
    with open(corpus_path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            mfl = line.replace("\\n", "\n")
            try:
                entries.append({"mfl": mfl, "statements": convert(parse(mfl))})
            except Exception as e:  # noqa: BLE001 — the class name is the datum
                entries.append({"mfl": mfl, "error": type(e).__name__})
    return entries


def reference_model(data_path):
    df = pd.read_csv(data_path)
    df["SEX"] = (df["ID"].astype(int) % 2).astype(int)
    tmp = "/tmp/mfl_anchor_data.csv"
    df.to_csv(tmp, index=False)
    m = create_basic_pk_model("oral", dataset_path=tmp)
    m = set_peripheral_compartments(m, 1)
    m = set_covariates(m, ["WT", "CRCL", "SEX"])
    di = m.datainfo
    sex = di["SEX"]
    di = di.set_column(sex.replace(variable_mapping=DataVariable.create("SEX", type="covariate", scale="nominal")))
    return m.replace(datainfo=di)


def dump_expand(model):
    cases = []
    for s in EXPAND_CASES:
        try:
            mf = ModelFeatures.create_from_mfl_string(s).expand(model)
            tuples = []
            for c in mf.covariate:
                for p in c.parameter:
                    for cov in c.covariate:
                        for fp in c.fp:
                            tuples.append([p, cov, fp, c.op, c.optional.option])
            cases.append({"mfl": s, "covariates": tuples})
        except Exception as e:  # noqa: BLE001
            cases.append({"mfl": s, "error": type(e).__name__})
    return {
        "model": {
            "pk": list(get_pk_parameters(model)),
            "iiv": list(get_individual_parameters(model, level="iiv")),
            "absorption": list(get_pk_parameters(model, "absorption")),
            "elimination": list(get_pk_parameters(model, "elimination")),
            "distribution": list(get_pk_parameters(model, "distribution")),
            "continuous": [
                c.name
                for c in model.datainfo
                if c.type == "covariate" and not c.variable.is_categorical()
            ],
            "categorical": [
                c.name
                for c in model.datainfo
                if c.type == "covariate" and c.variable.is_categorical()
            ],
        },
        "cases": cases,
    }


def main(corpus_path, data_path, out_path):
    doc = {
        "pharmpy_version": pharmpy.__version__,
        "parse": dump_parse(corpus_path),
        "expand": dump_expand(reference_model(data_path)),
    }
    with open(out_path, "w") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    print(f"pharmpy {pharmpy.__version__}: {len(doc['parse'])} parse entries, "
          f"{len(doc['expand']['cases'])} expand cases -> {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
