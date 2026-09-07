"""Dump Pharmpy's `modelsearch` candidate enumeration as JSON (#1181).

Runs *inside* the Pharmpy container — see run.sh. For every case in CASES
the script builds Pharmpy's base model (`create_base_model`, its least number
of transformations onto the space), filters the space against it
(`filter_mfl_statements`), builds the algorithm's workflow **without fitting
anything**, and walks the task graph:

* ``funcs``: the candidate moves, in Pharmpy's dictionary order;
* ``base_features``: ``get_model_features`` of the base;
* ``candidates``: one entry per ``create_candidate*`` task with its model
  name, the feature it applies (a list for `exhaustive`), the feature set of
  every candidate task upstream of it plus its own, and the model names of
  the nearest candidate ancestors — through a `choose_best_model` collector,
  which is how `reduced_stepwise` joins a group.

The output is committed at
crates/ferx-tools/tests/data/modelsearch_pharmpy_anchor.json and read by
crates/ferx-tools/src/modelsearch/pharmpy_anchor.rs.
"""

import json
import sys

import pharmpy
from pharmpy.modeling import create_basic_pk_model
from pharmpy.tools.mfl.helpers import key_to_str
from pharmpy.tools.mfl.parse import get_model_features, parse as mfl_parse
from pharmpy.tools.modelsearch import algorithms
from pharmpy.tools.modelsearch.tool import create_base_model, filter_mfl_statements
from pharmpy.workflows import ModelEntry

# (base model kind, MFL space, algorithm)
CASES = [
    ("oral", "ABSORPTION(FO); PERIPHERALS(0..1); LAGTIME([OFF,ON])", "exhaustive_stepwise"),
    (
        "oral",
        "ABSORPTION([INST,FO]); PERIPHERALS(0..1); LAGTIME([OFF,ON])",
        "exhaustive_stepwise",
    ),
    ("oral", "ABSORPTION(FO); PERIPHERALS(1..2)", "exhaustive_stepwise"),
    (
        "oral",
        "ABSORPTION([INST,FO]); PERIPHERALS(0..2); LAGTIME([OFF,ON])",
        "reduced_stepwise",
    ),
    ("oral", "ABSORPTION(FO); PERIPHERALS(0..2); LAGTIME([OFF,ON])", "reduced_stepwise"),
    ("oral", "ABSORPTION(FO); PERIPHERALS(0..2); LAGTIME([OFF,ON])", "exhaustive"),
    ("oral", "ABSORPTION([INST,FO]); PERIPHERALS(0..1); LAGTIME([OFF,ON])", "exhaustive"),
    ("oral", "PERIPHERALS(0..2); LAGTIME([OFF,ON])", "exhaustive_stepwise"),
    ("iv", "ABSORPTION([INST,FO]); LAGTIME([OFF,ON])", "exhaustive_stepwise"),
]


def is_candidate(task):
    fn = getattr(task.function, "__name__", "")
    return fn.startswith("create_candidate")


def is_collector(task):
    return getattr(task.function, "__name__", "") == "get_best_model"


def feature_of(task):
    """The feature(s) a candidate task applies, as key_to_str."""
    arg = task.task_input[1]
    if is_candidate(task) and isinstance(arg, tuple) and arg and isinstance(arg[0], tuple):
        # exhaustive: a combo of keys
        return [key_to_str(k) for k in arg]
    return [key_to_str(arg)]


def nearest_candidates(wf, task):
    """Model names of the nearest candidate ancestors, through fit tasks and
    collectors."""
    out = []
    stack = list(wf.get_predecessors(task))
    seen = set()
    while stack:
        t = stack.pop()
        if id(t) in seen:
            continue
        seen.add(id(t))
        if is_candidate(t):
            out.append(t.task_input[0])
        else:
            stack.extend(wf.get_predecessors(t))
    return sorted(out)


def dump_case(kind, space, algorithm):
    model = create_basic_pk_model(kind)
    mfl = mfl_parse(space, mfl_class=True)
    base = create_base_model(mfl, None, ModelEntry.create(model))
    funcs = filter_mfl_statements(mfl, base)
    wf, _ = getattr(algorithms, algorithm)(funcs, "no_add")
    candidates = []
    for task in wf.tasks:
        if not is_candidate(task):
            continue
        upstream = [t for t in wf.get_upstream_tasks(task) if is_candidate(t)]
        feature_set = set()
        for t in upstream:
            feature_set.update(feature_of(t))
        feature_set.update(feature_of(task))
        candidates.append(
            {
                "name": task.task_input[0],
                "feature": feature_of(task),
                "feature_set": sorted(feature_set),
                "parents": nearest_candidates(wf, task),
                "via_collector": any(is_collector(p) for p in wf.get_predecessors(task)),
            }
        )
    return {
        "model": kind,
        "space": space,
        "algorithm": algorithm,
        "input_features": get_model_features(model),
        "base_features": get_model_features(base.model),
        "base_transformations": base.model.description,
        "funcs": [key_to_str(k) for k in funcs],
        "candidates": candidates,
    }


def main(out):
    cases = [dump_case(*c) for c in CASES]
    with open(out, "w") as fh:
        json.dump({"pharmpy_version": pharmpy.__version__, "cases": cases}, fh, indent=1)
        fh.write("\n")
    print(f"wrote {out}: {len(cases)} cases")


if __name__ == "__main__":
    main(sys.argv[1])
