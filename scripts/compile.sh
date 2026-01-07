#!/bin/bash
echo "Compiling new version..."
nvcc -O3 -use_fast_math -lineinfo -Xptxas="-O3" -arch=sm_86 -o main.exe main.cu
# -maxregcount=40 might need tuning