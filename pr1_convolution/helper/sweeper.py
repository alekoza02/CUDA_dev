import subprocess
from time import perf_counter
from itertools import combinations

with open("eta_CLI_tracker.txt", "w") as f:
    print("", end="", file=f)

# Values to test
nice_numbers = [2, 4, 5, 8, 10, 15, 16, 20, 32, 50, 64, 100, 128, 200, 256, 400, 512]

def get_combinations(values):
    pairs = list(combinations(values, 2))
    return pairs

kc_values = get_combinations(nice_numbers[:-3]) # Stops at 200
block_values = get_combinations(nice_numbers) # Stops at 512

# Number of times to run each configuration
N_RUNS = 5

program_start = perf_counter()
best_min = 1e6
best_config = ""
iteration = 0

print(f"Will have {(len(kc_values) * len(block_values))} combinations")

for KCW_val, KCH_val in kc_values:
    for BX, BY in block_values:
        iteration += 1
        if KCW_val * KCH_val < 16000 and BX * BY < 1024:

            # Update config.h
            with open("../src/config.h", "w") as f:
                f.write(f"#pragma once\n")
                f.write(f"#define KCW {KCW_val}\n")
                f.write(f"#define KCH {KCH_val}\n")
                f.write(f"#define BLOCK_X {BX}\n")
                f.write(f"#define BLOCK_Y {BY}\n")

            # Compile
            subprocess.run([
                "nvcc",
                "-O3",
                "-use_fast_math",
                "-lineinfo",
                '-Xptxas=-O3',
                "-arch=sm_86",
                "-o", "../src/main.exe",
                "../src/main.cu"
            ], check=True)

            # Run and average
            total_time = 0
            skip_bc_long = False

            for i in range(N_RUNS):
                if not skip_bc_long:
                    result = subprocess.run(["../src/main.exe"], capture_output=True, text=True)
                    # Parse time in ms
                    for line in result.stdout.splitlines():
                        if "Done in" in line:
                            time_ms = int(line.split("Done in")[1].split("ms")[0].strip())
                            if time_ms > 100:
                                skip_bc_long = True
                            total_time += time_ms
                            print(f"Run {i+1}: {time_ms} ms")
                
                            break

            avg_time = total_time / N_RUNS
            
            if avg_time < best_min and not skip_bc_long:
                best_min = avg_time
                best_config = f"KCW={KCW_val}, KCH={KCH_val}, BLOCK=({BX},{BY})"

            with open("eta_CLI_tracker.txt", "a") as f:
                print(f"{perf_counter() - program_start}\t{100 * (iteration) / (len(kc_values) * len(block_values))}", file=f)

            print(f"\r [{100 * (iteration) / (len(kc_values) * len(block_values)):.6f}%] Best time: {best_min:.3f}ms, Best config: {best_config}")
