# llama.cpp Runtime Optimization Research

**Hardware:** Intel Xeon Platinum 8581C (Sapphire Rapids), 96 physical cores, 2 NUMA nodes, 188.9 GiB RAM, CPU-only  
**Model:** Meta-Llama-3.1-8B-Instruct-Q4_K_M (~4.5 GB)  
**ISA:** AMX, AVX-512 VNNI/BF16/VBMI all enabled in build flags  
**OS:** Linux 6.8 on GCP  
**Date:** 2026-04-30

---

## Status of Previously Documented Fixes

These were listed as open in `bottleneck-analysis.md` but are **already applied** in this codebase:

| Fix | File | Status |
|---|---|---|
| NUMA mbind(MPOL_INTERLEAVE) | `src/llama-mmap.cpp:472–477` | ✓ **Applied** — syscall present, uses nodemask=3 |
| GET_ROWS multi-thread | `ggml/src/ggml-cpu/ggml-cpu.c:2296` | ✓ **Applied** — parallelized with size guard |
| Tile config caching | `ggml/src/ggml-cpu/amx/mmq.cpp:204–224` | ✓ **Already optimal** — `thread_local` guard |
| ROPE parallelization | `ggml/src/ggml-cpu/ggml-cpu.c:2312–2316` | ✓ **Already optimal** — `n_tasks = n_threads` |
| Flash attention dispatch | `ggml/src/ggml-cpu/ggml-cpu.c:2351` | ✓ **Parallelized** — `n_tasks = n_threads` |
| MADV_HUGEPAGE for model weights | `src/llama-mmap.cpp:478–481` | ✓ **Applied** — after mbind, inside `#ifdef __linux__` |

The bottleneck-analysis.md summary table is **out of date** — fixes #1 and #2 are done.

---

## Remaining Optimization Opportunities

---

### Opportunity A: Transparent Huge Pages for Model Weights

#### Problem

The mmap loader calls `posix_madvise(POSIX_MADV_RANDOM)` when NUMA is enabled
([src/llama-mmap.cpp:462](src/llama-mmap.cpp)) but never calls `madvise(MADV_HUGEPAGE)`.

A 4.5 GB Q4_K_M model in 4KB pages requires ~1.15 million TLB entries. Sapphire Rapids
has 1024 L1 DTLB entries and 2048 L2 STLB entries — the entire model cannot be held in
TLB simultaneously, causing constant TLB miss → page-walk → memory stall inside every
GEMM kernel call.

With 2MB huge pages the same model needs only ~2250 TLB entries — well within the L2 STLB.

#### Evidence

No `MADV_HUGEPAGE` appears anywhere in `src/llama-mmap.cpp`. Confirm TLB pressure:

```bash
sudo perf stat -e dTLB-load-misses,dTLB-loads \
  ./build/bin/llama-bench \
  -m /home/ubuntu/diana/models/llama/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  -p 512 -n 128 --numa distribute -r 3
```

If `dTLB-load-misses / dTLB-loads > 5%`, TLB pressure is significant.

#### Hardware-Specific?

**Yes — especially beneficial on Sapphire Rapids.** The 8581C has 1.5× the L1 TLB
capacity of older Xeons but the model is proportionally larger. Multi-socket NUMA
makes page-walks more expensive (remote NUMA walk latency ~120 ns vs ~40 ns local).

#### Fix

In `src/llama-mmap.cpp`, after the existing `mbind()` call (line 477), add inside the
`#ifdef __linux__` block:

```cpp
// 2MB huge pages reduce TLB pressure from ~1.15M entries to ~2250 for a 4.5GB model.
// MADV_HUGEPAGE requests khugepaged to promote pages asynchronously; effective within
// the first few inference calls.
if (madvise(addr, file->size(), MADV_HUGEPAGE) != 0) {
    LLAMA_LOG_WARN("warning: madvise(MADV_HUGEPAGE) failed: %s\n", strerror(errno));
}
```

This requires Linux `MADV_HUGEPAGE` (kernel ≥ 2.6.38, confirmed on your 6.8 kernel).
No library dependency. No impact on non-Linux builds.

**Alternative — system-wide (no code change, test first):**

```bash
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo defer+madvise | sudo tee /sys/kernel/mm/transparent_hugepage/defrag
```

#### Expected Improvement

| Phase | Expected gain |
|---|---|
| Prefill PP512 (AMX-heavy) | 5–12% |
| Token generation TG128 | 3–8% |
| Combined with existing NUMA mbind | Multiplicative — total ~28–50% vs pre-fix baseline |

#### How to Measure

```bash
# Baseline TLB stats:
sudo perf stat -e dTLB-load-misses,dTLB-loads,iTLB-load-misses \
  ./build/bin/llama-bench \
  -m /home/ubuntu/diana/models/llama/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  -p 512 -n 128 --numa distribute -r 3 \
  > /dev/null

# After applying MADV_HUGEPAGE, verify promotion:
grep -i hugepages /proc/meminfo
# AnonHugePages should increase by ~4500 (pages promoted)

# Rerun same perf stat — dTLB-load-misses should drop by >80%
```

---

### Opportunity B: KV Cache Quantization (Q8_0 / Q4_0) via Flash Attention

#### Problem

KV cache defaults to F16 (`src/llama-context.cpp:2909–2910`). For Llama 3.1-8B:

- 32 layers, 32 KV heads, head dim = 128, F16 = 2 bytes
- Per-token KV size: `32 × 32 × 128 × 2 × 2 = 524 KB`
- At 512-token context: **256 MB** of KV data read every decode step

This is bandwidth that must traverse the memory bus every generated token. At 188 GB/s
effective bandwidth, 256 MB = 1.4 ms per token from KV alone.

Q8_0 would halve this to 128 MB, and Sapphire Rapids' L3 cache (96 MB total) could hold
~75% of a 512-token KV cache warm — eliminating most of the DRAM bandwidth cost.

#### Constraint

Quantized V cache requires flash attention (`src/llama-context.cpp:352–353`, `2986`):

```
quantized V cache was requested, but this requires Flash Attention
```

Flash attention is available on CPU via `-fa` / `--flash-attn on`.

Q8_0 K cache is compatible with flash attention with no extra constraint (line 2964 only
checks block size alignment, which Q8_0 satisfies for head_dim=128).

#### Hardware-Specific?

**Yes.** Sapphire Rapids benefits disproportionately:
- 96-core L3 = 96 MB shared — exactly the right size to absorb a Q8_0 KV cache at typical context lengths
- AVX-512 VNNI provides fast int8 dequant for Q8_0 at decode time
- Without quantization, memory bus is the decode bottleneck at long contexts

#### Model-Specific?

Partially. Q8_0 KV introduces ~0.1–0.3% perplexity degradation. Acceptable for most
generation tasks; may not be acceptable for reasoning chains or code generation where
precision matters. Test with `llama-perplexity` before deploying.

#### Fix — Enable at Runtime (No Code Change)

```bash
# Enable flash attention + Q8_0 KV cache (both K and V):
./build/bin/llama-bench \
  -m /home/ubuntu/diana/models/llama/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  -p 512 -n 128 --numa distribute \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  -t 64 -r 5 -o json \
  > results/llama_bench_q8_kv.json
```

Note: `--cache-type-v q8_0` requires `--flash-attn on` — without FA it will error.

#### Expected Improvement

| Context length | Expected TG gain |
|---|---|
| 128 tokens (warm-up) | 2–5% (KV cache small, mostly compute-bound) |
| 512 tokens | 8–15% |
| 2048 tokens | 15–25% (KV dominates bandwidth) |
| 8192 tokens | 25–40% |

The gain scales with context length because KV cache bandwidth is linear in N while
compute is constant per new token.

#### How to Measure

```bash
# Compare baseline vs Q8_0 KV at multiple context lengths:
for CTX in 128 512 2048; do
  ./build/bin/llama-bench \
    -m /home/ubuntu/diana/models/llama/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    -p $CTX -n 128 --numa distribute \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
    -t 64 -r 3
done

# Verify accuracy is not degraded:
./build/bin/llama-perplexity \
  -m /home/ubuntu/diana/models/llama/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  -f /path/to/wikitext-2-raw-v1.txt
```

---

### Opportunity C: Separate Thread Counts for Prefill vs Decode

#### Problem

Prefill (prompt processing) and decode (token generation) have fundamentally different
parallelism profiles:

- **Prefill** (PP): matrix multiply over B×N tokens — highly parallel, scales to 64–96 threads
- **Decode** (TG): single-token step — low parallelism, plateaus at ~32 threads (observed in `bottleneck-analysis.md` table: 32→96 threads = +4% only)

Using the same thread count (64) for both wastes 32 idle cores during decode, but more
importantly the synchronization overhead of 64 threads for single-token work adds latency
at every barrier.

#### Evidence

From `bottleneck-analysis.md` thread scaling table:

| Threads | TG t/s | Speedup |
|---|---|---|
| 32 | 36.58 | — |
| 96 | 37.89 | +1.04× |

96 threads is barely faster than 32 for TG. Optimal TG thread count is likely 16–32.

#### Fix — Runtime Flag (No Code Change)

`--threads-batch` (`common/arg.cpp:1124`) sets threads for prefill; `--threads` sets
threads for decode. They are independent.

```bash
# Recommended starting point:
./build/bin/llama-bench \
  -m /home/ubuntu/diana/models/llama/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  -p 512 -n 128 --numa distribute \
  --threads 32 --threads-batch 64 \
  -r 5 -o json \
  > results/llama_bench_thread_split.json
```

#### Expected Improvement

| Metric | Expected gain |
|---|---|
| TG t/s (decode) | 5–12% (reduced barrier overhead) |
| PP t/s (prefill) | 0–3% (already near-optimal at 64) |
| Time-to-first-token | 0–3% |

The gain on TG comes from thread barrier cost: with 64 threads, each barrier synchronizes
64 threads across 2 NUMA nodes (inter-socket barrier = ~500 ns × N threads). Halving to
32 reduces both barrier count and inter-socket traffic.

#### How to Measure

```bash
# Sweep thread count for decode:
for T in 8 16 24 32 48 64; do
  echo "=== --threads $T ==="
  ./build/bin/llama-bench \
    -m /home/ubuntu/diana/models/llama/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    -p 1 -n 128 --numa distribute \   # p=1 isolates decode
    --threads $T -r 3
done
# Find the inflection point (usually 16–32 for 8B models)
```

---

### Opportunity D: Microbatch Size Tuning (`--ubatch-size`)

#### Problem

`--ubatch-size` (default 512) controls how many tokens are processed per GEMM call
during prefill. On Sapphire Rapids, the AMX tile GEMM processes 16×16 BF16 tiles.
Optimal ubatch depends on:

1. **L3 fit**: AMX needs rows of `weight_matrix` (K × N fp16 = 4096 × 11008 × 2 = 90 MB
   for the FFN) in L3. This doesn't change with ubatch. But the activation matrix
   (ubatch × K) grows linearly.
2. **Thread granularity**: With 64 threads and ubatch=512, each thread handles 8 rows.
   With ubatch=256, each thread handles 4 rows — may be below the AMX tile minimum.
3. **Barrier frequency**: Smaller ubatch = more barriers per prompt.

The sweet spot for 96-core Sapphire Rapids is empirically ~256–512 for prefill.

#### Fix — Runtime Flag (No Code Change)

```bash
# Sweep ubatch sizes:
for UB in 64 128 256 512 1024; do
  echo "=== --ubatch-size $UB ==="
  ./build/bin/llama-bench \
    -m /home/ubuntu/diana/models/llama/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    -p 512 -n 0 --numa distribute \   # n=0: prefill only
    --threads-batch 64 --ubatch-size $UB -r 3
done
```

#### Expected Improvement

3–8% on prefill throughput depending on current default efficiency. Unlikely to change
decode (ubatch has no effect on single-token TG).

---

### Opportunity E: BPE Tokenizer Heap Allocations (Server Workloads)

**Status: still open.** Documented in `bottleneck-analysis.md` as Bottleneck #3.  
Relevant for the server with many short requests, not for offline benchmarks.

See `src/llama-vocab.cpp:614–616, 689–704` for the fix location.

---

### Opportunity F: Grammar O(n²) Deduplication

**Status: still open.** Documented in `bottleneck-analysis.md` as Bottleneck #4.  
Only relevant when `--grammar-file` / JSON schema constrained generation is used.

See `src/llama-grammar.cpp:868, 880, 1388` for the fix location.

---

## Recommended Action Plan

Apply in this order — each is independently measurable and buildable.

### Step 1 — Zero-code: thread and cache tuning (measure today)

No rebuild required. These are runtime flags that can be tested immediately:

```bash
# Baseline (current):
./build/bin/llama-bench \
  -m /home/ubuntu/diana/models/llama/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  -p 512 -n 128 --numa distribute -t 64 -r 5 -o json \
  > results/baseline_llama31.json

# Candidate A: thread split + Q8_0 KV + flash attention:
./build/bin/llama-bench \
  -m /home/ubuntu/diana/models/llama/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  -p 512 -n 128 --numa distribute \
  --threads 32 --threads-batch 64 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  -r 5 -o json \
  > results/candidate_a.json

python3 scripts/compare-llama-bench.py results/baseline_llama31.json results/candidate_a.json
```

Expected gain from flags alone: **10–25% TG**, **0–5% TTFT**.

### Step 2 — One-line code fix: MADV_HUGEPAGE

In `src/llama-mmap.cpp`, inside the existing `#ifdef __linux__` block at line ~477
(after the `mbind()` syscall), add:

```cpp
if (madvise(addr, file->size(), MADV_HUGEPAGE) != 0) {
    LLAMA_LOG_WARN("warning: madvise(MADV_HUGEPAGE) failed: %s\n", strerror(errno));
}
```

Rebuild, rerun benchmark:

```bash
cmake --build build --target llama-bench -j$(nproc)

./build/bin/llama-bench \
  -m /home/ubuntu/diana/models/llama/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  -p 512 -n 128 --numa distribute \
  --threads 32 --threads-batch 64 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  -r 5 -o json \
  > results/candidate_b_hugepages.json

# Verify huge pages promoted:
grep AnonHugePages /proc/meminfo

# Confirm TLB pressure drop:
sudo perf stat -e dTLB-load-misses,dTLB-loads \
  ./build/bin/llama-bench \
  -m /home/ubuntu/diana/models/llama/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  -p 512 -n 128 --numa distribute -t 32 -r 1 > /dev/null
```

Expected **additional** gain on top of Step 1: **5–12%**.

### Step 3 — If server latency matters: fix tokenizer heap allocations

Apply the `memcmp`-based stale-bigram check described in `bottleneck-analysis.md`
Bottleneck #3 to `src/llama-vocab.cpp:614–616`.

Measure with `llama-server` under load, not `llama-bench`.

---

## Cumulative Expected Improvement Summary

| Configuration | PP512 t/s | TG128 t/s | TTFT |
|---|---|---|---|
| Baseline (64 threads, F16 KV, no FA) | ~200 | ~34 | ~2.5 s |
| + thread split (32/64) + Q8_0 KV + FA | +0–5% | **+10–25%** | +0–5% |
| + MADV_HUGEPAGE | **+5–12%** | +3–8% | +5–12% |
| **Combined total** | **+5–17%** | **+13–33%** | **+5–17%** |

Note: gains are multiplicative, not additive. The table shows incremental gains at each step.  
Ranges reflect uncertainty in baseline configuration and THP promotion timing.

---

## What Would Not Help Here

- **More threads beyond 64 for decode** — profiling shows plateau at 32. Adding threads adds
  barrier latency, not throughput.
- **AVX-512 BF16 attention kernels** — attention is ~10–15% of compute at 512-token context;
  the AMX path already handles the dominant Q4_K_M GEMM. Reward/complexity ratio is low.
- **Prefetch intrinsics in Q4_K_M vec-dot** — the AMX batch GEMM (`mmq.cpp`) handles the
  hot path for prefill. The generic `ggml_vec_dot_q4_K_q8_K` is only called for small
  batch sizes. Benefit is <3%.
- **Per-NUMA-node work queues** — the mbind fix already distributes pages; per-NUMA queues
  would add ~2–5% on top but require significant scheduler refactoring.
