#!/bin/bash
log=$({ time ./main.exe; } 2>&1)
time_ms=$(echo "$log" | grep -oP 'Done in \K[0-9]+')

echo "Program took $i: ${time_ms} ms"