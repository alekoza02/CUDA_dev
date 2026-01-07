#!/bin/bash

# Default N = 10
N=${1:-10}
sum=0

echo "Running $N iterations + 1 of warmup"
echo "---"

for ((i=0; i<=N+1; i++)); do
  # Capture real time from `time`
  log=$({ time ./main.exe; } 2>&1)

  # Extract elapsed time (seconds, possibly fractional)
  time_ms=$(echo "$log" | grep -oP 'Done in \K[0-9]+')

  if [ "$i" -eq 0 ]; then
    echo "Iteration $i: ${time_ms} ms [SKIPPED — warmup]"
    echo "---"
    continue
  fi

  echo "Iteration $i: ${time_ms} ms"
  sum=$(awk "BEGIN {print $sum + $time_ms}")
done

mean=$(awk "BEGIN {print $sum / ($N + 1)}")
echo "Average time over $N runs: ${mean} ms"
