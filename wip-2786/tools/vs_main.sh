#!/bin/bash
# vs_main.sh <example> <data>: OFV + estimates, main binary vs the current incremental (path off).
W=/root/package/ferx-core/.worktrees/lanes
M="$1"; case "$M" in /*) ;; *) M="$W/examples/$M";; esac
for b in /scratch/mu/ad/ferx-main-0bda4d0a /scratch/mu/ad/ferx-kjit-inc; do
  r=$(mktemp -d /scratch/mu/ad/kj/vm.XXXX); cp "$M" "$r/m.ferx"
  (cd "$r" && timeout 1200 "$b" m.ferx --data "$2" > out.txt 2>&1)
  echo "$(basename $b): $(grep -m1 -E '^OFV' $r/out.txt)"
  grep -E '^ +[A-Z_]+[A-Z0-9_]* += +[-0-9.e]+' "$r/out.txt" | head -12 > "/scratch/mu/ad/kj/vm-$(basename $b).txt"
  rm -rf "$r"
done
diff -q /scratch/mu/ad/kj/vm-ferx-main-0bda4d0a.txt /scratch/mu/ad/kj/vm-ferx-kjit-inc.txt >/dev/null && echo "estimates identical" || diff /scratch/mu/ad/kj/vm-ferx-main-0bda4d0a.txt /scratch/mu/ad/kj/vm-ferx-kjit-inc.txt | head
