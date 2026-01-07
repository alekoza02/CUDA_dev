# Objective

Manage 4096x4096 image and 200x200 kernel

# Progress

Using a 1024x1024 image and a 200x200 kernel ---> mean of 20

Step name | Time [ms]
--- | ---
Raw kernel | 112.571
Tiling kernel | 108.952
Best params selection | 80.143

# Details

1) Raw kernel
    - Launch of normal kernel with no optimizations: just pure application of the algorithm 
    - noted from the profiling `1_baseline.ncu-rep`:
        - Excessive global ---> L1/TEX cache usage
    - ideas:
        - Try maybe using shared memory
        - Read about halo kernels launch
        - Sliding window

2) Tiling kernel
    - Splitting the kernel into smaller chunks and loading one by one into the GPU as \_\_constant\_\_ memory
    - noted from the profiling `2_tiling_kernel.ncu-rep`:
        - Compute (SM) [\%] goes from 100% to 60%
        - Memory [\%] goes from 100% to 48%
        - Constant memory usage explodes (from 0 to 40.5M)
        - Drop down global ---> L1/TEX by -87.50% 
        - Extremly high Stalls in:
            - MIO Throttle
            - Short Scoreboard
        - Extremely low Stalls in:
            - LG Throttle
            - Long Scoreboard
    - ideas:
        - Maybe launch overhead is bigger than the benefit of shared memory? Should try with starting kernel 100x100 and multiply result by 4 ---> 41.48 * 4 = 165ms
        - Try using shared memory for image
        - Play with launch settings and kernel sizes `KCW` and `KCH`

3) Best params selection
    - After running `sweeper.py`, new best params are:
        ```c++
        #pragma once
        #define KCW 50
        #define KCH 25
        #define BLOCK_X 16
        #define BLOCK_Y 16
        ```
    - noted from the profiling `3_best_params.ncu-rep`:
        - Compute (SM) [\%] goes almost to 100% again
        - Memory [\%] goes almost to 100% again
        - Twice the LSU, but no more ADU
        - Thrice the Eligible Warps per Scheduler
        - Extremely high Stalls in:
            - LG Throttle
            - Twice the IMC Miss
        - Extremely low Stalls in:
            - MIO Throttle
            - Short Scoreboard
