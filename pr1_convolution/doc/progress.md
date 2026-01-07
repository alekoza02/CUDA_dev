# Objective

Manage 4096x4096 image and 200x200 kernel

# Progress

Using a 1024x1024 image and a 200x200 kernel ---> mean of 20

Step name | Time [ms]
--- | ---
1) Raw kernel | 112.571

# Details

1) Raw kernel
    - Launch of normal kernel with no optimizations: just pure application of the algorithm 
    - noted from the profiling `baseline.ncu`:
        - Excessive global ---> L1/TEX cache usage
    - ideas:
        - Try maybe using shared memory
        - Read about halo kernels launch
        - Sliding window