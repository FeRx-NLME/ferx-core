#!/bin/bash
# Usage (from worktree root): bash profile.sh <ferx-binary> <model> <data> <outfile> [sample_seconds] [delay]
BIN=$1; M=$2; D=$3; OUTF=$4; SECS=${5:-4}; DELAY=${6:-0.5}
"$BIN" "$M" --data "$D" --clean > /dev/null 2>&1 &
PID=$!
sleep "$DELAY"
sample "$PID" "$SECS" -mayDie -file "$OUTF" > /dev/null 2>&1
wait "$PID"
echo "done pid=$PID"
