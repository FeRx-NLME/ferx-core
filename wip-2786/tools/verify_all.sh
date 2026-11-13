#!/bin/bash
# Rebuild; regression fits (estimates off vs on); covariance comparisons; instruction counts.
J=8 bash /scratch/mu/ad/kj/build_inc.sh || exit 1
echo "===== regress"
bash /scratch/mu/ad/kj/regress.sh 2>&1 | grep -E '^###|identical|^[<>]|== on' | grep -v '== off'
echo "===== cov"
bash /scratch/mu/ad/kj/covall.sh 2>&1 | grep -E '^###|norm-wise|== on'
echo "===== instructions"
bash /scratch/mu/ad/kj/instr3_all.sh 2>&1 | grep -v build
