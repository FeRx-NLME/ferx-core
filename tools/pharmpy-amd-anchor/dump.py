"""Dump Pharmpy's `amd` sequencing and search-space split (#1184).

What AMD itself owns is the *orchestration*: which tools run, in what order,
and which statements of one search space each of them is handed. The numerics
belong to the individual tools and are anchored with them (#1180-#1183). So
this dumps exactly those two things, without fitting anything:

* `get_subtool_order(strategy)` for every strategy Pharmpy offers, and
* for a corpus of AMD-shaped MFL spaces, the subspace Pharmpy hands its
  structural search (`get_search_space_modelsearch`) and its covariate search
  (`get_search_space_covsearch`), plus the ALLOMETRY statement it lifts out.

Usage: python3 dump.py <corpus.txt> <out.json>
"""

import json
import sys

import pharmpy
from pharmpy.tools.amd.run import (
    ALLOWED_STRATEGY,
    get_search_space_covsearch,
    get_search_space_modelsearch,
    get_subtool_order,
)
from pharmpy.tools.mfl.parse import parse as mfl_parse
from pharmpy.tools.mfl.stringify import stringify


def spaces(corpus):
    for line in corpus:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ss = mfl_parse(line, True)
        allometry = ss.allometry
        yield {
            "mfl": line,
            # The raw partition: what the space itself says about each tool.
            "pk": str(ss.filter("pk")),
            "covariate": stringify(ss.covariate) if ss.covariate else "",
            "allometry": (
                {"covariate": allometry.covariate, "reference": allometry.reference}
                if allometry is not None
                else None
            ),
            # What `amd` actually hands the tool, its defaults filled in.
            "modelsearch": str(get_search_space_modelsearch(ss, "basic_pk", "oral")),
            "covsearch": str(get_search_space_covsearch(ss, "basic_pk", "oral")),
        }


def main(corpus_path, out_path):
    with open(corpus_path) as f:
        corpus = list(f)
    out = {
        "pharmpy_version": pharmpy.__version__,
        "orders": {s: get_subtool_order(s) for s in ALLOWED_STRATEGY},
        "spaces": list(spaces(corpus)),
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")
    print(
        f"wrote {out_path}: {len(out['spaces'])} spaces, Pharmpy {out['pharmpy_version']}"
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
