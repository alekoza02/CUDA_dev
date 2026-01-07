# Objective

Manage 4096x4096 image and 200x200 kernel

# Progress

Using a 1024x1024 image and a 200x200 kernel ---> mean of 20

Step name | Time [ms]
--- | ---
1) Raw kernel | 112.571
2) Tiling kernel | 108.952

# Details

1) Raw kernel
    - Launch of normal kernel with no optimizations: just pure application of the algorithm 
    - noted from the profiling `baseline.ncu-rep`:
        - Excessive global ---> L1/TEX cache usage
    - ideas:
        - Try maybe using shared memory
        - Read about halo kernels launch
        - Sliding window

2) Tiling kernel
    - Splitting the kernel into smaller chunks and loading one by one into the GPU as \_\_constant\_\_ memory
    - noted from the profiling `tiling_kernel.ncu-rep`:
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