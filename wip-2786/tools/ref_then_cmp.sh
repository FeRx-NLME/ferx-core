#!/bin/bash
# ref_then_cmp.sh: build the merge-base reference (once), then the branch and the comparison.
[ -x /scratch/mu/ad/ferx-main-feat ] || J=6 bash /scratch/mu/ad/kj/build_ref.sh || exit 1
bash /scratch/mu/ad/kj/vsmain.sh
