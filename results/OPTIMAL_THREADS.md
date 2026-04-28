# Optimal Thread Count — Benchmark Analysis

**OPTIMAL_THREADS=96**

Hardware: Intel Xeon Platinum 8581C (96 vCPUs) — same machine used for profiler run.

## Prompt Processing (PP) — avg_ts (tokens/sec), higher is better

| threads | pp128  | pp512  | pp1024 | pp2048 | mean   |
|---------|--------|--------|--------|--------|--------|
| 1       | 16.01  | 17.48  | 16.69  | 15.58  | 16.44  |
| 8       | 98.39  | 100.61 | 99.39  | 96.63  | 98.76  |
| 16      | 186.03 | 180.82 | 171.87 | 168.79 | 176.88 |
| 32      | 237.95 | 235.44 | 230.33 | 222.21 | 231.48 |
| **96**  | **272.10** | **303.62** | **297.03** | **286.86** | **289.90** |

## Token Generation (TG, n_gen=128) — avg_ts (tokens/sec)

| threads | b=512  | b=2048 |
|---------|--------|--------|
| 1       | 2.24   | 2.19   |
| 8       | 15.47  | 15.60  |
| 16      | 25.52  | 25.43  |
| 32      | 37.14  | 36.03  |
| **96**  | **40.17** | 35.61  |

## Rationale

- **PP**: t=96 wins decisively across all prompt sizes (~25% faster than t=32).
- **TG**: t=96 wins at batch=512 (40.17 vs 37.14, +8%). At batch=2048 it is
  within 1% of t=32 (35.61 vs 36.03) — effectively a tie.
- PP dominates wall-clock time in typical inference workloads, so optimizing for
  PP throughput is the correct choice.
- For profiling (next task), t=96 is also the most informative thread count since
  it is where threading overhead (libgomp spin-wait, synchronization) becomes
  most visible in `perf` callgraphs.

## Profiler Run

**Command:**
```bash
./run_profile.sh -t 96 /home/ubuntu/diana/models/llama/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

**OPTIMAL_THREADS used:** 96

### Top Hotspots (from `results/perf_flat.txt`)

| Overhead | Library               | Symbol                                      |
|----------|-----------------------|----------------------------------------------|
| 33.87%   | libgomp.so            | spin-wait loop (thread synchronisation)      |
| 19.51%   | libggml-cpu.so        | `tinygemm_kernel_vnni` (Q4_K)                |
| 11.81%   | libggml-cpu.so        | `tinygemm_kernel_amx` (Q4_K)                 |
|  8.00%   | libggml-cpu.so        | `tinygemm_kernel_vnni` (Q6_K)                |
|  5.11%   | libgomp.so            | spin-wait loop                               |
|  4.63%   | libgomp.so            | spin-wait loop                               |
|  2.62%   | libggml-cpu.so        | `tinygemm_kernel_amx` (Q6_K)                 |
|  2.21%   | libggml-cpu.so        | `tinyBLAS::gemm_bloc<4,6>`                   |
|  1.98%   | libgomp.so            | spin-wait loop                               |
|  1.41%   | libggml-cpu.so        | `ggml_cpu_fp32_to_fp16`                      |

### Key Observations

- **~46% libgomp spin-wait**: At t=96, the majority of CPU time is spent in
  OpenMP thread synchronisation barriers — expected on a 96-vCPU machine where
  not all threads have useful work every scheduling round.
- **~42% GEMM kernels**: The actual compute is dominated by AMX/VNNI tinygemm
  kernels for Q4_K and Q6_K quantisation formats, called via
  `ggml_backend_amx_mul_mat`.
- **tinyBLAS + fp16 conversion** account for ~3.6% — secondary hot paths for
  smaller matmuls and type conversion.
