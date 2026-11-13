#!/bin/bash
# The kjit end-of-fit counters per model.
B=${B:-/scratch/mu/ad/ferx-kjit-cf}
S=/scratch/speed/bench
for m in ${MODELS:-pk1@pk1_n500 pk2@pk2_n300}; do
  name=${m%@*}; data=${m#*@}
  r=$(mktemp -d /scratch/mu/ad/kj/sv.XXXX); cp "$S/$name.ferx" "$r/m.ferx"
  echo "## $name"
  (cd "$r" && FERX_KJIT=1 FERX_KJIT_DEBUG=1 timeout 600 "$B" m.ferx --data "$S/$data.csv" 2>&1 | grep -E 'kjit: (compiled [0-9]|validation|subject not|[0-9]+ distinct)' | sort | uniq -c | head -4)
  rm -rf "$r"
done
