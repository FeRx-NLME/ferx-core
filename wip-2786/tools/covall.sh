#!/bin/bash
cd /scratch/mu/ad/kj
D=/root/package/ferx-core/.worktrees/lanes/data
echo "### warfarin"; ./covcmp.sh warfarin.ferx $D/warfarin.csv 2>&1 | tail -3
echo "### two_cpt_oral_cov"; ./covcmp.sh two_cpt_oral_cov.ferx $D/two_cpt_oral_cov.csv 2>&1 | tail -3
echo "### mm_oral"; ./covcmp.sh mm_oral.ferx $D/mm_oral.csv 2>&1 | tail -3
echo "### warfarin_ode"; ./covcmp.sh warfarin_ode.ferx $D/warfarin.csv 2>&1 | tail -3
