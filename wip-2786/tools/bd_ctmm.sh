#!/bin/bash
J=6 bash /scratch/mu/ad/kj/build_inc.sh > /scratch/mu/ad/kj/build.log 2>&1
echo build $?
tail -5 /scratch/mu/ad/kj/build.log
timeout 900 bash /scratch/mu/ad/kj/dbg_raw.sh ${1:-ctmm_2state.ferx} 2>&1 | sort | uniq -c | sort -rn | head -40
