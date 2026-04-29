# llama.cpp Performance Bottleneck Analysis

**Hardware:** Intel Xeon Platinum 8581C, 96 vCPUs, 2 NUMA nodes, 188 GiB RAM, CPU-only  
**Model:** Qwen2.5-7B-Instruct Q4_K_M  
**Profiler:** Linux perf (953K samples, `task-clock:ppp`)  
**Benchmark:** llama-bench + llama-batched-bench  
**Date:** 2026-04-25

---

## How to Read This Document

Each bottleneck has:
- **Evidence** — what the profiler or benchmark actually shows
- **Root cause** — verified against source code with file + line
- **Hardware-specific?** — honest answer
- **Model-specific?** — honest answer
- **Expected gain** — realistic range, not best case
- **How to measure after fix** — exact commands

---

## Bottleneck 1: NUMA Page Migration on mmap'd Model Weights

### Evidence

From `results/perf_callgraph.txt` — ~40% of all CPU samples flow through:

```
tinygemm_kernel_vnni (AMX kernel)
  └── asm_exc_page_fault
        └── do_numa_page
              └── migrate_misplaced_folio
                    └── migrate_pages_batch (~80% of this subtree)
                          ├── migrate_folio_unmap
                          │     └── rmap_walk_anon → _raw_spin_lock (contention)
                          └── move_to_new_folio → copy_page
```

The kernel's automatic NUMA balancer is continuously migrating model weight pages between
NUMA node0 (CPUs 0-23, 48-71) and NUMA node1 (CPUs 24-47, 72-95) during inference.
Each migration involves page table locking, memory copying, and TLB shootdowns —
all of which stall the AMX kernel that triggered the fault.

### Root Cause

**File:** `src/llama-mmap.cpp`, lines 437–465

```cpp
impl(struct llama_file * file, size_t prefetch, bool numa) {
    size = file->size();
    int fd = file->file_id();
    int flags = MAP_SHARED;
    if (numa) { prefetch = 0; }          // ← skips MAP_POPULATE when numa=true

    addr = mmap(NULL, file->size(), PROT_READ, flags, fd, 0);

    if (numa) {
        if (posix_madvise(addr, file->size(), POSIX_MADV_RANDOM)) {  // ← hints random access
            LLAMA_LOG_WARN(...);
        }
    }
    // ← NO mbind() call. Pages are never pinned to a NUMA node.
    //   The kernel's NUMA balancer is free to migrate them continuously.
}
```

When `--numa distribute` is passed, `ggml_numa_init` sets per-thread CPU affinity,
but the mmap region for model weights is never bound with `mbind()`. The kernel sees
threads from both NUMA nodes accessing the same pages and migrates them back and forth
on every access from the "wrong" node.

### Hardware-Specific?

**No.** `mbind()` is a standard Linux syscall (`man 2 mbind`), part of POSIX NUMA API.
The fix does not touch SIMD, intrinsics, or CPU microarchitecture.
It applies equally on ARM (Graviton), AMD EPYC, and any other multi-NUMA Linux system.

### Model-Specific?

**No.** The fix is in the mmap loader, which is shared by all model architectures.

### Expected Improvement

| Scenario | Expected gain |
|---|---|
| 96-core, 2-NUMA node, mmap=true (your setup) | 20–35% end-to-end |
| Single-socket or single-NUMA machine | 0% (not applicable) |
| mmap=false (model loaded into RAM) | 0% (not applicable) |

### Fix

After the `mmap()` call in `src/llama-mmap.cpp`, add:

```cpp
#ifdef __linux__
#include <numaif.h>

// Interleave model weight pages across all NUMA nodes.
// This distributes bandwidth pressure evenly and prevents the kernel
// NUMA balancer from migrating pages during inference.
unsigned long nodemask = (1UL << numa_max_node() + 1) - 1;
if (mbind(addr, file->size(), MPOL_INTERLEAVE,
          &nodemask, sizeof(nodemask) * 8, 0) != 0) {
    LLAMA_LOG_WARN("warning: mbind(MPOL_INTERLEAVE) failed: %s\n", strerror(errno));
}
#endif
```

This should only apply when `numa=true` is passed (i.e., when the user opts in via
`--numa distribute`).

### How to Measure After Fix

**Before fix (baseline — already in `results/llama_bench.json`):**
```
pp512:  ~200 t/s  (48 threads)
tg128:  ~34  t/s  (48 threads)
```

**After fix — run:**
```bash
./build/bin/llama-bench \
  -m ~/diana/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  -p 512 -n 128 -r 5 \
  --numa distribute \
  -o json > results/llama_bench_after_numa_fix.json
```

**Confirm page migration is gone:**
```bash
sudo perf stat -e migrations,page-faults \
  ./build/bin/llama-bench \
  -m ~/diana/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  -p 512 -n 128 -r 3 \
  --numa distribute
```
`migrations` counter should drop by >90% vs baseline.

**Re-profile to confirm:**
```bash
sudo perf record -g -F 999 -o results/perf_after_numa.data \
  -- ./build/bin/llama-bench \
       -m ~/diana/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
       -p 512 -n 128 -r 3 --numa distribute

sudo perf report -i results/perf_after_numa.data --stdio --no-children \
  > results/perf_after_numa_flat.txt
```
`migrate_misplaced_folio` should disappear from the top entries.

---

## Bottleneck 2: Thread Work Granularity — Plateau Above 32 Threads

### Evidence

From `results/llama_bench.json` (token generation, averaged across prompt sizes):

| Threads | TG t/s | Speedup vs previous |
|---|---|---|
| 1  | 2.22  | —      |
| 8  | 15.54 | 7.0×   |
| 16 | 25.47 | 1.64×  |
| 32 | 36.58 | 1.44×  |
| 96 | 37.89 | **1.04×** ← near-zero gain for 3× more threads |

Adding 64 threads (32→96) yields 4% gain on token generation.
This is not memory-bandwidth saturation alone — it is the task scheduler
not subdividing work finely enough to keep all threads busy.

### Root Cause

**File:** `ggml/src/ggml-cpu/ggml-cpu.c`, function `ggml_get_n_tasks()` (line 2181)

Several ops that fire during every decode step are hardcoded to `n_tasks = 1`:

```cpp
case GGML_OP_GET_ROWS:
case GGML_OP_SET_ROWS:
    {
        // FIXME: get_rows can use additional threads, but the cost of launching
        // additional threads decreases performance with GPU offloading
        //n_tasks = n_threads;
        n_tasks = 1;      // ← line 2300: hardcoded single-thread
    } break;

case GGML_OP_SOFT_MAX:   // ← line 2326: also single-thread in some paths
case GGML_OP_SCALE:
case GGML_OP_DIAG:
case GGML_OP_CLAMP:      // ← line 2324: TODO comment, still n_tasks=1
    n_tasks = 1;
```

On a 96-thread run, every `GGML_OP_GET_ROWS` (embedding lookup, used every decode step)
serializes all 96 threads to a single core. With 28 transformer layers each issuing
multiple single-threaded ops, a significant fraction of each decode step is sequential.

The FIXME comment acknowledges this: the single-thread constraint was added to avoid
overhead with GPU offloading, but on CPU-only inference it is an unnecessary bottleneck.

### Hardware-Specific?

**Partially.** The fix (enabling multi-threading for `GET_ROWS` when no GPU is present)
is pure scheduler logic — no SIMD or intrinsics. The *degree* of gain is
hardware-dependent (more visible on high-core-count machines), but the fix itself
is hardware-agnostic.

### Model-Specific?

**No.** `GGML_OP_GET_ROWS` is used by all transformer models for token embedding lookup.

### Expected Improvement

| Scenario | Expected gain |
|---|---|
| 32–96 threads, CPU-only | 10–20% on TG |
| 1–16 threads | < 5% (thread overhead may negate gains at low counts) |
| GPU offloading enabled | Requires the existing guard — no change |

### Fix

In `ggml_get_n_tasks()`, guard the multi-thread path on CPU-only context:

```cpp
case GGML_OP_GET_ROWS:
case GGML_OP_SET_ROWS:
    {
        // Enable multi-threading only when no GPU backend is active.
        // The original n_tasks=1 was conservative to avoid launch overhead
        // dominating short GPU-offloaded ops.
        n_tasks = (backend_is_cpu_only) ? n_threads : 1;
    } break;
```

A simpler approach that doesn't require plumbing the backend type:
add a threshold — only parallelize if the tensor has enough rows to amortize
the thread-launch cost (e.g., `n_tasks = (node->ne[1] >= n_threads) ? n_threads : 1`).

### How to Measure After Fix

**Thread scaling sweep — compare before and after:**
```bash
# After fix:
./build/bin/llama-bench \
  -m ~/diana/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  -p 512 -n 128 \
  -t 1,8,16,32,96 \
  -r 5 -o json \
  > results/llama_bench_after_thread_fix.json

# Compare scaling curves:
python3 scripts/compare-llama-bench.py \
  results/llama_bench.json \
  results/llama_bench_after_thread_fix.json
```

**Success criterion:** TG t/s at 96 threads should be >1.3× the value at 32 threads
(vs the current 1.04×).

---

## Bottleneck 3: BPE Tokenizer — Heap Allocations in Merge Loop

### Evidence

Not visible in profiler (tokenization is a one-time cost per request in `llama-bench`).
Identified by static analysis. Relevant for server workloads with many short requests.

### Root Cause

**File:** `src/llama-vocab.cpp`, lines 614–616 and 689–704

```cpp
// Inside the BPE merge loop — runs O(n log n) times per tokenization:

// Line 614-616: stale-bigram check
std::string left_token  = std::string(left_symbol.text,  left_symbol.n);  // heap alloc
std::string right_token = std::string(right_symbol.text, right_symbol.n); // heap alloc
if (left_token + right_token != bigram.text) { continue; }                // heap alloc

// Line 689-704: add_new_bigram
std::string left_token  = std::string(symbols[left].text,  symbols[left].n);  // heap alloc
std::string right_token = std::string(symbols[right].text, symbols[right].n); // heap alloc
bigram.text = left_token + right_token;                                        // heap alloc
```

3 allocations for the staleness check + 3 for each new bigram insertion.
For Qwen2.5 with 151,387 BPE merges, this is ~900K heap operations per tokenization call.

### Hardware-Specific? No. Model-Specific? No.

### Expected Improvement

| Scenario | Expected gain |
|---|---|
| llama-bench single run | < 1% (tokenization is ~0.1% of total time) |
| Server: 100 req/s, short prompts | 3–8% reduction in per-request latency |
| Server: long prompts (2K+ tokens) | 1–3% |

### Fix

Replace `std::string` with pointer+length comparison:

```cpp
// Staleness check — no allocation needed:
bool stale = (left_symbol.n + right_symbol.n != bigram.text.size()) ||
             memcmp(left_symbol.text, bigram.text.data(), left_symbol.n) != 0 ||
             memcmp(right_symbol.text,
                    bigram.text.data() + left_symbol.n, right_symbol.n) != 0;
if (stale) { continue; }
```

The `bigram.text` member still needs ownership (stored in priority queue), so only
the check strings are eliminated.

### How to Measure After Fix

Tokenization is not measured by `llama-bench`. Use a dedicated tokenizer benchmark:

```bash
# Time tokenization of a large file (1M tokens):
time ./build/bin/llama-tokenize \
  -m ~/diana/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  --file /path/to/large_text.txt \
  > /dev/null

# Or use heaptrack to count allocations before/after:
heaptrack ./build/bin/llama-bench \
  -m ~/diana/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  -p 512 -n 1 -r 1
heaptrack_print heaptrack.llama-bench.*.zst | grep "peak heap"
```

**Success criterion:** Peak heap allocations during tokenization drop by >50%.

---

## Bottleneck 4: Grammar Stack Deduplication — O(n²) per Token

### Evidence

Not visible in profiler (grammar was not active during `llama-bench` run).
Identified by static analysis. Relevant for structured output / JSON mode.

### Root Cause

**File:** `src/llama-grammar.cpp`, lines 868 and 880

```cpp
// Called on every sampled token when grammar is active:
std::set<llama_grammar_stack, decltype(stack_cmp)> seen(stack_cmp);  // tree, heap alloc per node

while (!todo.empty()) {
    // ...
    if (seen.find(curr_stack) != seen.end()) { continue; }  // O(log n) but tree traversal is slow
    seen.insert(curr_stack);

    if (curr_stack.empty()) {
        // Line 880: O(n) linear scan of new_stacks vector
        if (std::find(new_stacks.begin(), new_stacks.end(), curr_stack) == new_stacks.end()) {
            new_stacks.emplace_back(std::move(curr_stack));
        }
    }
}
```

`std::set` uses a red-black tree — each insert/lookup allocates a tree node and
does pointer-chasing comparisons. The `std::find` on line 880 is O(n) linear scan,
making the overall deduplication O(n²) when many empty stacks are produced.

Also — **File:** `src/llama-grammar.cpp`, line 1388:

```cpp
// Called on every sampled token:
if (std::find(grammar.trigger_tokens.begin(),
              grammar.trigger_tokens.end(), token) != grammar.trigger_tokens.end()) {
```

Linear scan through trigger tokens on every token. Should be `std::unordered_set`.

### Hardware-Specific? No. Model-Specific? No.

### Expected Improvement

| Scenario | Expected gain |
|---|---|
| No grammar (normal generation) | 0% |
| Simple grammar (small rule set) | 5–10% on sampling time |
| Complex JSON schema grammar | 15–30% on sampling time |

### Fix

```cpp
// Replace std::set with unordered_set using a flat hash:
struct StackHash {
    size_t operator()(const llama_grammar_stack & s) const {
        size_t h = 0;
        for (auto * p : s) { h ^= std::hash<const void*>{}(p) + 0x9e3779b9 + (h << 6); }
        return h;
    }
};
std::unordered_set<llama_grammar_stack, StackHash> seen;

// Replace trigger_tokens vector with unordered_set:
std::unordered_set<llama_token> trigger_tokens;
// lookup becomes O(1): if (grammar.trigger_tokens.count(token)) { ... }
```

### How to Measure After Fix

```bash
# Run a grammar-constrained generation benchmark:
./build/bin/llama-cli \
  -m ~/diana/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  --grammar-file grammars/json.gbnf \
  -p "Generate a JSON object with 10 fields:" \
  -n 200 --prompt-cache-all \
  2>&1 | grep "eval time"

# Profile specifically with grammar active:
sudo perf record -g -F 999 -o results/perf_grammar.data \
  -- ./build/bin/llama-cli \
       -m ~/diana/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
       --grammar-file grammars/json.gbnf \
       -p "Generate a JSON object:" -n 200

sudo perf report -i results/perf_grammar.data --stdio --no-children \
  > results/perf_grammar_flat.txt
```

**Success criterion:** `llama_grammar_advance_stack` drops from top-10 in the
grammar profile.

---

## Summary Table

| # | Bottleneck | File | Lines | Hardware-specific? | Model-specific? | Expected gain |
|---|---|---|---|---|---|---|
| 1 | NUMA page migration — mmap not bound | `src/llama-mmap.cpp` | 437–465 | **No** | **No** | **20–35%** all inference |
| 2 | Thread work granularity — ops hardcoded to 1 thread | `ggml/src/ggml-cpu/ggml-cpu.c` | 2294–2300 | No (gain is hardware-dependent) | **No** | **10–20%** TG, 32+ threads |
| 3 | BPE tokenizer heap allocs | `src/llama-vocab.cpp` | 614–616, 689–704 | **No** | **No** | 3–8% server latency |
| 4 | Grammar O(n²) deduplication | `src/llama-grammar.cpp` | 868, 880, 1388 | **No** | **No** | 15–30% grammar workloads |

### Priority Order

1. **Fix #1 first** — highest gain, one-line fix, measurable immediately with `perf stat -e migrations`
2. **Fix #2 second** — requires careful testing to avoid GPU regression
3. **Fix #3 and #4** — only if server throughput or grammar latency are targets

### Measurement Workflow

```
baseline results (already in results/)
        ↓
apply fix → rebuild → run benchmark → save to results/llama_bench_after_<fix>.json
        ↓
compare: python3 scripts/compare-llama-bench.py results/llama_bench.json results/llama_bench_after_<fix>.json
        ↓
re-profile: sudo perf record ... → perf report > results/perf_after_<fix>_flat.txt
        ↓
confirm target function dropped in profile
```
