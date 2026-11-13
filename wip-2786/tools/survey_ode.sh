#!/bin/bash
# survey_ode.sh: value-read survey on ODE example models (current incremental binary).
W=/root/package/ferx-core/.worktrees/lanes
export B=/scratch/mu/ad/ferx-kjit-inc
export CASES="${CASES:-$W/examples/warfarin_ode.ferx@$W/data/warfarin.csv $W/examples/two_cpt_iv_ode.ferx@$W/data/two_cpt_iv.csv}"
bash /scratch/mu/ad/kj/survey.sh 2>&1 | cut -c1-700
