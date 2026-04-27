# llama.cpp Optimisation Report

## Context

| Field | Value |
|-------|-------|
| **Machine** | `diana-c4-highcpu` (`c4-highcpu-96-amir`) |
| **CPU** | Intel Xeon Platinum 8581C @ 2.30 GHz, 48 cores / 96 threads, 2 NUMA nodes |
| **L3 Cache** | 260 MiB (shared) |
| **RAM** | 188 GiB DDR |
| **OS** | Ubuntu 22.04, kernel 6.8.0-1053-gcp |
| **Model** | Qwen2.5-7B-Instruct Q4_K_M (4.36 GiB, 7.62 B params) |
| **Build** | llama.cpp commit `c2de5dfaa` (build 8782) |
| **Date** | 2026-04-23 |

> **Note:** The benchmarking model used for all collected runs was **Qwen2.5-7B-Instruct Q4_K_M**.
> The final `run_bench.sh` script was configured for **Llama 3.1-8B-Instruct Q4_K_M** with
> extended parameter sweeps (ubatch, KV cache types, flash attention) and build-level flags
> (`-march=native -O3`), but this script had not yet been executed at the time of report
> generation. The data below reflects all runs actually committed to `results/llama_bench.json`.

---

## Full Comparison Table — Baseline vs Best Optimised

The baseline is the **llama-bench default** thread count (`-t 8`) with default batch
parameters (`-b 512 -ub 512`), f16 KV cache, flash attention disabled.
The best-optimised configuration uses 96 threads (all vCPUs) with the same batch/cache
settings, which produced the highest throughput.

### Prompt Processing (pp) — tokens/sec

| n_prompt | Baseline (t=8) | Best (t=96) | Δ t/s | Δ % |
|---------:|---------------:|------------:|------:|----:|
| 128 | 98.39 | 272.10 | +173.71 | +176.6% |
| 512 | 100.61 | 303.62 | +203.01 | +201.8% |
| 1024 | 99.39 | 297.03 | +197.64 | +198.9% |
| 2048 | 96.63 | 286.86 | +190.23 | +196.9% |

### Text Generation (tg 128 tokens) — tokens/sec

| Metric | Baseline (t=8) | Best (t=96) | Δ t/s | Δ % |
|--------|---------------:|------------:|------:|----:|
| tg 128 | 15.47 | 40.17 | +24.70 | +159.6% |

### Summary

| | pp 512 (t/s) | tg 128 (t/s) |
|---|---:|---:|
| **Baseline** (t=8, b=512, ub=512, KV=f16, FA=off) | 100.61 | 15.47 |
| **Best Optimised** (t=96, b=512, ub=512, KV=f16, FA=off) | 303.62 | 40.17 |
| **Speedup** | **3.02×** | **2.60×** |

---

## Per-Optimisation Breakdown

### 1. Thread Count Tuning

Thread count had by far the largest impact. The machine has 48 physical cores (96
hyperthreads across 2 NUMA nodes). Throughput scaled well up to 32 threads and continued
to improve (with diminishing returns) to 96.

| Threads | pp 512 (t/s) | Δ pp vs prev | tg 128 (t/s) | Δ tg vs prev |
|--------:|-------------:|-------------:|--------------:|-------------:|
| 1 | 17.48 | — | 2.24 | — |
| 8 | 100.61 | +83.13 (+475.6%) | 15.47 | +13.23 (+590.6%) |
| 16 | 180.82 | +80.21 (+79.7%) | 25.52 | +10.05 (+64.9%) |
| 32 | 235.44 | +54.62 (+30.2%) | 37.14 | +11.62 (+45.5%) |
| 96 | 303.62 | +68.18 (+29.0%) | 40.17 | +3.03 (+8.2%) |

**Verdict:** Best at `-t 96` (all vCPUs). Prompt processing scales nearly linearly to 32 threads;
text generation saturates around 32–48 threads due to the sequential, memory-bound nature of
autoregressive decoding.

### 2. Batch Size (`-b`) Sweep

Two batch sizes were tested: 512 (default) and 2048. The micro-batch size (`-ub`) remained
fixed at 512 throughout.

| Config (t=96) | pp 512 (t/s) | tg 128 (t/s) |
|---------------|-------------:|--------------:|
| b=512, ub=512 | 303.62 | 40.17 |
| b=2048, ub=512 | 293.15 | 35.61 |
| **Δ** | **−10.47 (−3.5%)** | **−4.56 (−11.3%)** |

At t=32 the picture is mixed (b=2048 gives pp +3.80 t/s but tg −1.11 t/s).

**Verdict:** Increasing `-b` to 2048 while keeping `-ub 512` provided **no benefit** at the
optimal thread count. The default `-b 512` is preferred. The full ubatch sweep (`-ub 128,256,512,1024`)
configured in `run_bench.sh` was not yet executed; finer ubatch tuning may yet yield gains.

### 3. KV Cache Quantisation (`q8_0` vs `f16`)

**Not evaluated in collected data.** All runs used `--cache-type-k f16 --cache-type-v f16`.
The `run_bench.sh` script includes a sweep over `q8_0` and `f16` for both K and V caches.
Expected effect: `q8_0` reduces KV memory by ~50%, which can improve throughput at longer
contexts by reducing memory bandwidth pressure, at the cost of minor precision loss.

**Verdict:** Delta unknown — pending execution of `run_bench.sh`.

### 4. Flash Attention (`--flash-attn`)

**Not evaluated in collected data.** All runs had `flash_attn=false`. The `run_bench.sh`
script includes a dedicated flash-attention sweep (Run 2) with `--flash-attn 1` using f16 KV.
Note from `batched_bench.txt`: the batched bench context log shows `flash_attn = auto` resolved
to **enabled**, confirming the build supports flash attention on this CPU.

**Verdict:** Delta unknown — pending execution of `run_bench.sh`.

### 5. NUMA-Aware Launch + OpenMP Environment Variables

**Not separately evaluated.** The existing benchmark data was collected before the NUMA/OpenMP
tuning was added to `run_bench.sh`. The script now includes:

```bash
numactl --localalloc              # pin memory to local NUMA node
export OMP_WAIT_POLICY=passive    # yield instead of spin-wait
export OMP_PROC_BIND=close        # keep threads on nearby cores
export OMP_PLACES=cores           # one OMP thread per physical core
```

These settings directly target the **33.87% libgomp spin-wait overhead** observed in
`perf_flat.txt` (see Bottlenecks section). `OMP_WAIT_POLICY=passive` is expected to
substantially reduce the idle-spin CPU waste, and `numactl --localalloc` should reduce
cross-socket memory latency on this 2-NUMA-node system.

**Verdict:** Delta unknown — pending execution of `run_bench.sh`. Expected to be significant
given the ~47% combined libgomp overhead in profiling.

### 6. Build-Level Flags (`-march=native -O3`)

**Not separately evaluated.** The existing data was produced from the default CMake Release
build. The `run_bench.sh` script now passes:

```cmake
-DCMAKE_CXX_FLAGS='-march=native -O3'
-DCMAKE_C_FLAGS='-march=native -O3'
```

`-march=native` on this Xeon Platinum 8581C enables AVX-512, AVX-VNNI, AMX-INT8, AMX-BF16,
and AVX512-FP16 instruction sets. The existing build already uses AMX/VNNI kernels (visible
in `perf_flat.txt`), so the incremental gain from `-march=native` may be modest if the
build system was already auto-detecting ISA extensions. `-O3` enables additional loop
optimisations and vectorisation beyond the default `-O2`.

**Verdict:** Delta unknown — pending rebuild and re-bench. Expected improvement is small
(0–5%) since the critical matmul kernels already use hand-written AMX/VNNI intrinsics.

---

## Recommended Final Configuration

The following command reproduces the best-performing configuration from the available data,
augmented with the NUMA/OpenMP tuning and build flags from `run_bench.sh`:

### Build

```bash
cmake -B build . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS='-march=native -O3' \
  -DCMAKE_C_FLAGS='-march=native -O3' \
  -DLLAMA_PERF=ON \
  -DGGML_PERF=ON

cmake --build build --config Release -j$(nproc) --target llama-bench
```

### Runtime Environment

```bash
export OMP_WAIT_POLICY=passive
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

### Run

```bash
numactl --localalloc ./build/bin/llama-bench \
  -m /path/to/model.gguf \
  -t 96 \
  -b 512 \
  -ub 512 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --flash-attn 0 \
  -p 128,512,1024,2048 \
  -n 128 \
  -r 5
```

### Parameter Summary

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `-t` | 96 | All vCPUs; best pp and tg throughput |
| `-b` | 512 | Default; `-b 2048` showed no improvement |
| `-ub` | 512 | Default; further tuning pending |
| `--cache-type-k` | f16 | Default; `q8_0` sweep pending |
| `--cache-type-v` | f16 | Default; `q8_0` sweep pending |
| `--flash-attn` | 0 | Sweep pending; build supports FA |
| `OMP_WAIT_POLICY` | passive | Reduces libgomp spin-wait overhead |
| `OMP_PROC_BIND` | close | Improves cache locality |
| `OMP_PLACES` | cores | One thread per physical core |
| `numactl` | `--localalloc` | Keeps allocations on local NUMA node |
| CMake CXX/C flags | `-march=native -O3` | Enables full ISA (AVX-512, AMX) + aggressive optimisation |

---

## Remaining Bottlenecks

Analysis based on `results/perf_flat.txt` (953 K samples of `task-clock:ppp`):

### 1. libgomp Spin-Wait — 46.85% of total CPU time

| Symbol (offset in libgomp.so) | Overhead |
|-------------------------------|----------|
| `0x00000000000207ba` | 33.87% |
| `0x00000000000207be` | 5.11% |
| `0x00000000000207c6` | 4.63% |
| `0x00000000000208ef` | 1.98% |
| `0x00000000000207b8` | 0.98% |
| `0x0000000000020602` | 0.36% |
| Other libgomp symbols | ~0.9% |
| **Total libgomp** | **~47%** |

These addresses correspond to the OpenMP barrier spin-loop in `libgomp`. Nearly half of all
CPU cycles are spent in idle spin-waiting for worker threads. This is the single largest
inefficiency.

**Mitigation (configured but not yet tested):**
- `OMP_WAIT_POLICY=passive` switches from spin-wait to yield/sleep, dramatically reducing
  wasted cycles at the cost of slightly higher wake-up latency.
- `OMP_PROC_BIND=close` + `OMP_PLACES=cores` reduces thread migration overhead.

### 2. Matrix Multiplication Kernels — ~44% of total CPU time

| Symbol | Library | Overhead |
|--------|---------|----------|
| `tinygemm_kernel_vnni<..., block_q4_K, ...>::apply` | libggml-cpu | 19.51% |
| `tinygemm_kernel_amx<..., block_q4_K, ...>` | libggml-cpu | 11.81% |
| `tinygemm_kernel_vnni<..., block_q6_K, ...>::apply` | libggml-cpu | 8.00% |
| `tinygemm_kernel_amx<..., block_q6_K, ...>` | libggml-cpu | 2.62% |
| `tinyBLAS<...>::gemm_bloc<4, 6>` | libggml-cpu | 2.21% |
| **Total matmul** | | **~44%** |

These are the productive compute kernels (AMX tile multiply and VNNI dot-product paths for
Q4_K and Q6_K quantised weights). This is expected and desirable — after removing the
spin-wait overhead, matmul should dominate.

**Further work:**
- The split between AMX and VNNI paths (e.g. 11.81% AMX vs 19.51% VNNI for Q4_K) suggests
  the AMX tile path is used for larger tiles while VNNI handles residuals. Ensuring prompt
  lengths are multiples of the AMX tile size (typically 256) could shift more work to the
  faster AMX path.
- `ggml_cpu_fp32_to_fp16` at 1.41% indicates non-trivial overhead in fp32→fp16 conversion,
  potentially reducible by keeping more intermediate results in fp16.

### 3. Kernel / NUMA Overhead — ~2%

| Symbol | Overhead |
|--------|----------|
| `__pv_queued_spin_lock_slowpath` | 0.56% |
| `down_read_trylock` | 0.43% |
| `up_read` | 0.23% |
| `do_user_addr_fault` | 0.14% |
| NUMA migration (`do_numa_page`, `task_numa_fault`, `migrate_misplaced_folio`, etc.) | ~0.5% |

NUMA page migration is visible in the kernel profile (`do_numa_page`, `numa_migrate_prep`,
`migrate_misplaced_folio`, `task_numa_placement`). While currently small (~0.5%), this could
grow at higher thread counts spanning both NUMA nodes.

**Mitigation:** `numactl --localalloc` (configured in `run_bench.sh`) should eliminate most
cross-node migration by pinning allocations to the local node.

### 4. Other Notable Symbols

| Symbol | Overhead | Note |
|--------|----------|------|
| `ggml_vec_dot_f16` | 0.93% | f16 dot product in attention |
| `quantize_row_q8_K_ref` | 0.61% | Dynamic quantisation for AMX input |
| `ggml_compute_forward_soft_max` | 0.28% | Softmax in attention |
| `ggml_compute_forward_set_rows` | 0.21% | Row scatter operation |
| `__tls_get_addr` / `@plt` | 0.58% | TLS overhead from shared libraries |

### Summary of Actionable Next Steps

1. **Run `./run_bench.sh`** with the Llama 3.1-8B model to collect the full parameter sweep
   (ubatch, KV quant, flash attention) with build-level and NUMA/OpenMP optimisations active.
2. **Re-run `perf`** after setting `OMP_WAIT_POLICY=passive` to verify the spin-wait overhead
   drops from ~47% to near zero.
3. **Evaluate q8_0 KV cache** — expected to improve long-context throughput by halving KV
   memory bandwidth.
4. **Evaluate flash attention** — should reduce attention memory footprint and may improve
   tg throughput at longer contexts.
5. **Consider `-t 48`** (physical cores only, no hyperthreads) with `OMP_PLACES=cores` — may
   reduce contention vs `-t 96` which uses both hyperthreads per core.
