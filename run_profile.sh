#!/usr/bin/env bash
# Usage: ./run_profile.sh /path/to/model.gguf
# Profiles llama-bench with perf (Linux) and saves call graph + flamegraph data.
# Requires: perf, and optionally flamegraph (https://github.com/brendangregg/FlameGraph)

set -e

MODEL=${1:?Usage: $0 /path/to/model.gguf}
REPO_DIR="$(cd "$(dirname "$0")/llama.cpp" && pwd)"
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
       -r 3

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
