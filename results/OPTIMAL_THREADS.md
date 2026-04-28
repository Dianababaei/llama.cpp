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
