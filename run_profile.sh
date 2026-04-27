#!/usr/bin/env bash
# Usage: ./run_profile.sh [-t <threads>] [/path/to/model.gguf]
# Profiles llama-bench with perf (Linux) and saves call graph + flamegraph data.
# Requires: perf, and optionally flamegraph (https://github.com/brendangregg/FlameGraph)
#
# Baseline run: ./run_profile.sh -t 96
# (optimal thread count = 96 = nproc, from baseline bench sweep)
#
# Baseline perf profile top 10 symbols (Qwen2.5-7B Q4_K_M, t=96):
#   1. 33.87%  libgomp     spin-loop (thread synchronisation overhead)
#   2. 19.51%  tinygemm_kernel_vnni<..., block_q4_K, ..., 64, 256>
#   3. 11.81%  tinygemm_kernel_amx<..., block_q4_K, ..., 256, 0>
#   4.  8.00%  tinygemm_kernel_vnni<..., block_q6_K, ..., 64, 256>
#   5.  5.11%  libgomp     spin-loop (offset 0x207be)
#   6.  4.63%  libgomp     spin-loop (offset 0x207c6)
#   7.  2.62%  tinygemm_kernel_amx<..., block_q6_K, ..., 256, 0>
#   8.  2.21%  tinyBLAS::gemm_bloc<4,6>  (FP16 matmul fallback)
#   9.  1.98%  libgomp     spin-loop (offset 0x208ef)
#  10.  1.41%  ggml_cpu_fp32_to_fp16
# Total libgomp overhead: ~46% — confirms multi-threaded execution captured.
# AMX/VNNI matmul kernels: ~42% — dominant compute.

set -e

# --- Parse optional flags ------------------------------------------------- #
THREADS=""
while getopts ":t:" opt; do
  case $opt in
    t) THREADS="$OPTARG" ;;
    \?)
      echo "Error: unrecognised option -$OPTARG" >&2
      echo "Usage: $0 [-t <threads>] /path/to/model.gguf" >&2
      exit 1
      ;;
    :)
      echo "Error: option -$OPTARG requires an argument" >&2
      echo "Usage: $0 [-t <threads>] /path/to/model.gguf" >&2
      exit 1
      ;;
  esac
done
shift $((OPTIND - 1))

MODEL=${1:-/home/ubuntu/diana/models/llama/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf}
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$REPO_DIR/build"
RESULTS_DIR="$(dirname "$0")/results"
mkdir -p "$RESULTS_DIR"

# Build with debug symbols (keeps Release speed, adds symbol info)
echo "=== [1/3] Building with symbols ==="
cmake -B "$BUILD_DIR" "$REPO_DIR" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLAMA_PERF=ON \
  -DGGML_PERF=ON
cmake --build "$BUILD_DIR" --config RelWithDebInfo -j"$(nproc)" \
  --target llama-bench

echo "=== [2/3] perf record (prompt processing + generation) ==="
sudo perf record \
  -g \
  -F 999 \
  -o "$RESULTS_DIR/perf.data" \
  -- "$BUILD_DIR/bin/llama-bench" \
       -m "$MODEL" \
       -p 512 \
       -n 128 \
       -r 3 \
       ${THREADS:+-t $THREADS}

echo "=== [3/3] Generating reports ==="

# Flat report: which functions cost the most
sudo perf report \
  -i "$RESULTS_DIR/perf.data" \
  --stdio \
  --no-children \
  > "$RESULTS_DIR/perf_flat.txt"

# Call graph report: who calls what
sudo perf report \
  -i "$RESULTS_DIR/perf.data" \
  --stdio \
  --call-graph fractal \
  > "$RESULTS_DIR/perf_callgraph.txt"

# Flamegraph (optional — skip if not installed)
if command -v stackcollapse-perf.pl &>/dev/null && command -v flamegraph.pl &>/dev/null; then
  sudo perf script -i "$RESULTS_DIR/perf.data" \
    | stackcollapse-perf.pl \
    | flamegraph.pl > "$RESULTS_DIR/flamegraph.svg"
  echo "  flamegraph.svg     — interactive flamegraph"
else
  # Save raw stacks anyway so flamegraph can be generated later
  sudo perf script -i "$RESULTS_DIR/perf.data" \
    > "$RESULTS_DIR/perf_stacks.txt"
  echo "  perf_stacks.txt    — raw stacks (run stackcollapse-perf.pl | flamegraph.pl to render)"
fi

echo ""
echo "Results saved to $RESULTS_DIR/"
echo "  perf_flat.txt      — top functions by CPU cost (start here)"
echo "  perf_callgraph.txt — full call graph"
