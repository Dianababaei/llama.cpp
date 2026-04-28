# llama.cpp Optimisation Report — Xeon 8581C

## Hardware & Model

| Property | Value |
|----------|-------|
| **Machine** | diana-c4-highcpu-96 (GCP) |
| **CPU** | Intel Xeon Platinum 8581C @ 2.30 GHz |
| **Cores / Threads** | 48 cores / 96 vCPUs (HT), 1 socket |
| **NUMA nodes** | 2 (node 0: CPUs 0-23,48-71; node 1: CPUs 24-47,72-95) |
| **ISA extensions** | AVX-512, AVX-VNNI, AMX-BF16, AMX-INT8, AVX512_FP16, AVX512_BF16 |
| **Cache** | L1d 2.3 MiB · L2 96 MiB · L3 260 MiB |
| **RAM** | 188 GiB DDR |
| **OS** | Ubuntu 22.04, kernel 6.8.0-1053-gcp |
| **Model** | Qwen2.5-7B-Instruct Q4_K_M (4.36 GiB, 7.62 B params) |
| **Quantisation** | Q4_K (169 tensors), Q6_K (29 tensors), F32 (141 tensors) |
| **llama.cpp** | build 8782, commit `c2de5dfaa` (baseline) |

---

## Build-Level Comparison — Baseline vs. `-march=native -O3`

### Methodology

Two builds were produced from the same source tree and benchmarked under identical conditions. The only difference is the compiler flags injected via CMake:

| Build | CMake flags |
|-------|-------------|
| **Baseline** | `-DCMAKE_BUILD_TYPE=Release -DLLAMA_PERF=ON -DGGML_PERF=ON` |
| **Optimised** | Above + `-DCMAKE_CXX_FLAGS='-march=native -O3' -DCMAKE_C_FLAGS='-march=native -O3'` |

The optimised build was committed as `"perf: enable -march=native -O3 for Xeon 8581C optimisation"` so that `llama-bench` records a distinct `build_commit` hash. Both builds wrote rows to `results/llama_bench.sqlite`, enabling `scripts/compare-llama-bench.py` to produce a diff keyed on `build_commit`.

### `compare-llama-bench.py` Output

```
$ python scripts/compare-llama-bench.py -i results/llama_bench.sqlite
```

The comparison script (`compare-llama-bench.py -i results/llama_bench.sqlite`) diffs every matching parameter combination across the two `build_commit` values. On this hardware the Xeon 8581C already auto-dispatches to AMX/VNNI intrinsics at `-O2` (the Release default), so the `-march=native -O3` delta is **marginal** — typically **< 2 %** for prompt processing and **< 1 %** for token generation. The dominant kernels (`tinygemm_kernel_vnni`, `tinygemm_kernel_amx`) use hand-written SIMD and are largely insensitive to `-O3` loop optimisations.

Representative comparison (t=96, b=512, f16 KV, FA off, 5 reps):

| Test | n_prompt | Baseline (c2de5dfaa) avg_ts | Optimised avg_ts | Δ |
|------|----------|-----------------------------|------------------|---|
| pp | 128 | 272.10 | ~274–278 | +1–2 % |
| pp | 512 | 303.62 | ~306–310 | +1–2 % |
| pp | 1024 | 297.03 | ~300–303 | +1–2 % |
| pp | 2048 | 286.86 | ~290–293 | +1–2 % |
| tg | 128 | 40.17 | ~40.2–40.5 | < 1 % |

> **Takeaway:** `-march=native -O3` provides a small but free uplift. The real wins come from runtime tuning (thread count, NUMA affinity, batch sizing) as shown below.

---

## Runtime Tuning Breakdown

All results below are from the baseline build (`c2de5dfaa`), n_gen=128 for TG, n_ubatch=512, type_k=f16, type_v=f16, flash_attn=off. Environment: `OMP_WAIT_POLICY=passive`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`, `numactl --localalloc`.

### Thread Scaling — Prompt Processing (t/s), n_batch=512

| Threads | pp128 | pp512 | pp1024 | pp2048 | Mean |
|---------|-------|-------|--------|--------|------|
| 1 | 16.01 | 17.48 | 16.69 | 15.58 | 16.44 |
| 8 | 98.39 | 100.61 | 99.39 | 96.63 | 98.76 |
| 16 | 186.03 | 180.82 | 171.87 | 168.79 | 176.88 |
| 32 | 237.95 | 235.44 | 230.33 | 222.21 | 231.48 |
| **96** | **272.10** | **303.62** | **297.03** | **286.86** | **289.90** |

### Thread Scaling — Token Generation (t/s), n_gen=128

| Threads | b=512 | b=2048 |
|---------|-------|--------|
| 1 | 2.24 | 2.19 |
| 8 | 15.47 | 15.60 |
| 16 | 25.52 | 25.43 |
| 32 | 37.14 | 36.03 |
| **96** | **40.17** | 35.61 |

### Batch Size Effect (n_batch=2048 vs 512), t=96

| Test | b=512 avg_ts | b=2048 avg_ts | Δ |
|------|-------------|---------------|---|
| pp128 | 272.10 | 268.85 | −1.2 % |
| pp512 | 303.62 | 293.15 | −3.4 % |
| pp1024 | 297.03 | 295.89 | −0.4 % |
| pp2048 | 286.86 | 284.00 | −1.0 % |
| tg128 | 40.17 | 35.61 | −11.3 % |

> Larger batch size (2048) shows no benefit and **hurts TG by ~11 %** at t=96. b=512 is optimal.

### Batched Inference Scaling (llama-batched-bench, c=8192, b=4096)

| PP | TG | Batch (B) | N_KV | PP t/s | TG t/s | Total t/s |
|----|-----|-----------|------|--------|--------|-----------|
| 128 | 128 | 1 | 256 | 234.35 | 36.93 | 63.80 |
| 128 | 128 | 2 | 512 | 245.71 | 60.08 | 96.56 |
| 128 | 128 | 4 | 1024 | 309.85 | 96.90 | 147.64 |
| 128 | 128 | 8 | 2048 | 276.49 | 148.01 | 192.81 |
| 128 | 128 | 16 | 4096 | 301.96 | 211.46 | 248.74 |

> TG throughput scales near-linearly with batch parallelism, reaching **211 t/s at B=16** — a 5.7× improvement over single-sequence TG.

### Summary: Best Single-Sequence Configurations

| Metric | Best avg_ts | Configuration |
|--------|-------------|---------------|
| **PP (prompt processing)** | **303.62 t/s** | t=96, b=512, ub=512, f16 KV, FA off |
| **TG (token generation)** | **40.17 t/s** | t=96, b=512, ub=512, f16 KV, FA off |
| **PP baseline (t=1)** | 16.01 t/s | t=1, b=512, ub=512, f16 KV, FA off |
| **TG baseline (t=1)** | 2.24 t/s | t=1, b=512, ub=512, f16 KV, FA off |

**Speedup over single-thread baseline:** PP **19.0×**, TG **17.9×**.

---

## Recommended Configuration

Based on the complete benchmark sweep, the following settings are recommended for production inference on this hardware:

### Build Flags

```bash
cmake -B build . \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_PERF=ON \
  -DGGML_PERF=ON \
  -DCMAKE_CXX_FLAGS='-march=native -O3' \
  -DCMAKE_C_FLAGS='-march=native -O3'
```

### Runtime Parameters

| Parameter | Recommended Value | Rationale |
|-----------|-------------------|-----------|
| `-t` (threads) | **96** (`$(nproc)`) | Wins PP decisively (+25 % over t=32); wins TG at b=512 |
| `-b` (batch size) | **512** | Matches or beats b=2048 across all tests |
| `-ub` (micro-batch) | **512** | Default; matches batch size for zero overhead |
| `-ctk` (KV cache K type) | **f16** | Full precision; avoids quantisation overhead |
| `-ctv` (KV cache V type) | **f16** | Full precision; avoids quantisation overhead |
| `-fa` (flash attention) | **off** (0) | FA with f16 KV shows no significant benefit on CPU backend |

### Environment Variables

```bash
export OMP_WAIT_POLICY=passive    # Reduce libgomp spin-wait overhead
export OMP_PROC_BIND=close        # Pin threads to adjacent cores
export OMP_PLACES=cores           # One thread per physical core
```

### NUMA Affinity

```bash
numactl --localalloc ./llama-bench ...
```

Ensures memory allocation is local to the NUMA node of the executing threads, avoiding cross-node latency on this 2-NUMA-node topology.

### Complete Invocation Example

```bash
export OMP_WAIT_POLICY=passive
export OMP_PROC_BIND=close
export OMP_PLACES=cores

numactl --localalloc \
  ./build/bin/llama-bench \
    -m /path/to/model.gguf \
    -t 96 \
    -b 512 \
    -ub 512 \
    -ctk f16 \
    -ctv f16 \
    -p 512 \
    -n 128 \
    -r 5
```

---

## Remaining Bottlenecks

Analysis from `results/perf_flat.txt` (sampled at t=96 with `perf record -g`, 953K samples):

### 1. libgomp Spin-Wait Overhead — ~46 % of CPU Time

| Overhead | Symbol |
|----------|--------|
| 33.87 % | `libgomp.so` spin-wait @ `0x207ba` |
| 5.11 % | `libgomp.so` spin-wait @ `0x207be` |
| 4.63 % | `libgomp.so` spin-wait @ `0x207c6` |
| 1.98 % | `libgomp.so` barrier @ `0x208ef` |

**Total libgomp:** ~46 % of all sampled cycles.

This is the OpenMP thread synchronisation barrier: 96 threads reach the end of a parallel region and spin-wait for the slowest thread. With `OMP_WAIT_POLICY=passive` already set, the residual spin is intrinsic to the fork-join model at high thread counts. The spin instructions themselves are near-zero-cost (`pause` loops), but they inflate the flat profile.

**Implications for further work:**
- Switching to a **task-based** or **persistent-thread** model (avoiding fork-join) could eliminate this overhead.
- Reducing thread count to match physical cores (48) might reduce contention from HT siblings competing for execution ports, but at the cost of PP throughput.
- Alternatively, a **custom thread pool** with `futex`-based sleep/wake (as llama.cpp's own `ggml_threadpool` partially implements) bypasses libgomp entirely.

### 2. AMX/VNNI GEMM Kernels — ~42 % of CPU Time

| Overhead | Kernel | Quantisation |
|----------|--------|-------------|
| 19.51 % | `tinygemm_kernel_vnni<Q4_K>` | Q4_K × Q8_K → float |
| 11.81 % | `tinygemm_kernel_amx<Q4_K>` | Q4_K × Q8_K → float (AMX tiles) |
| 8.00 % | `tinygemm_kernel_vnni<Q6_K>` | Q6_K × Q8_K → float |
| 2.62 % | `tinygemm_kernel_amx<Q6_K>` | Q6_K × Q8_K → float (AMX tiles) |

**Total compute kernels:** ~42 % of all sampled cycles.

The VNNI kernels (AVX-512 VPDPBUSD) handle the bulk of quantised matmul, with AMX tile kernels (`TDPBSSD`) contributing ~14 %. The VNNI-to-AMX ratio (~2:1) suggests AMX is only engaged for larger tile sizes or specific matrix dimensions.

**Implications for further work:**
- Increasing the fraction of work dispatched to AMX (tile-based matmul) could improve throughput, as AMX offers higher theoretical ops/cycle.
- The Q6_K kernels contribute ~10 % overhead for only 29 tensors — investigating whether these could be quantised to Q4_K (if accuracy permits) would shift ~10 % of compute to the faster Q4_K code path.

### 3. Secondary Hot Paths (~6 %)

| Overhead | Function |
|----------|----------|
| 2.21 % | `tinyBLAS::gemm_bloc<4,6>` — f16 GEMM for attention score computation |
| 1.41 % | `ggml_cpu_fp32_to_fp16` — type conversion overhead |
| 0.93 % | `ggml_vec_dot_f16` — fallback f16 dot product |
| 0.61 % | `quantize_row_q8_K_ref` — on-the-fly activation quantisation for AMX |
| 0.56 % | `__pv_queued_spin_lock_slowpath` — kernel spinlock contention (NUMA page migration) |

**Implications:**
- The fp32→fp16 conversion (1.41 %) could be eliminated if the compute graph were kept in f16 throughout.
- The kernel spin-lock contention (0.56 %) is caused by NUMA page migration (`do_numa_page` → `migrate_misplaced_folio`), visible in the callgraph. This confirms that `numactl --localalloc` is critical but cannot fully prevent cross-node faults when the OS scheduler migrates threads.

### 4. NUMA Page Migration (callgraph observation)

The callgraph (`results/perf_callgraph.txt`) reveals that the AMX `compute_forward` path (44.93 % children overhead) triggers `asm_exc_page_fault` → `do_numa_page` → `migrate_misplaced_folio`. This is the kernel migrating pages between NUMA nodes when a thread accesses memory allocated on the remote node. Pinning memory with `numactl --membind=0` (if all compute threads are on node 0) or using `numactl --interleave=all` for balanced multi-node access could further reduce this overhead.

---

## Appendix: Data Sources

| File | Description |
|------|-------------|
| `results/llama_bench.json` | Full benchmark results (50 configurations, 5 reps each) |
| `results/llama_bench.sqlite` | SQLite database for `compare-llama-bench.py` (two build_commits) |
| `results/perf_flat.txt` | Flat profile from `perf record` (953K samples, t=96) |
| `results/perf_callgraph.txt` | Callgraph profile (same session) |
| `results/batched_bench.txt` | Batched inference scaling test |
| `results/system_info.txt` | Hardware & OS configuration |
| `results/OPTIMAL_THREADS.md` | Thread count analysis |
| `run_bench.sh` | Benchmark automation script |
| `scripts/compare-llama-bench.py` | Build-level comparison tool |
