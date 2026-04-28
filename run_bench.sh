#!/usr/bin/env bash
# Usage: ./run_bench.sh /path/to/model.gguf
# Builds llama.cpp, runs llama-bench + batched-bench, saves JSON results.

set -e

MODEL=${1:?Usage: $0 /path/to/model.gguf}
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$REPO_DIR/build"
RESULTS_DIR="$REPO_DIR/results"
mkdir -p "$RESULTS_DIR"

echo "=== [1/4] Building ==="
cmake -B "$BUILD_DIR" "$REPO_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_PERF=ON \
  -DGGML_PERF=ON
cmake --build "$BUILD_DIR" --config Release -j"$(nproc)" \
  --target llama-bench llama-batched-bench

echo "=== [2/4] System info ==="
{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "model: $MODEL"
  lscpu
  free -h
  uname -a
} > "$RESULTS_DIR/system_info.txt"

echo "=== [3/4] llama-bench (prompt sizes x gen lengths) ==="
"$BUILD_DIR/bin/llama-bench" \
  -m "$MODEL" \
  -p 128,512,1024,2048 \
  -n 128 \
  -b 512,2048 \
  -ub 128,256,512,1024 \
  -t 1,8,16,32,"$(nproc)" \
  -ctk f16,q8_0 \
  -ctv f16,q8_0 \
  -r 5 \
  -o json -oe sql \
  > "$RESULTS_DIR/llama_bench.json" \
  2> >(sqlite3 "$RESULTS_DIR/llama_bench.sqlite")

echo "=== [4/4] batched-bench (parallelism scaling) ==="
"$BUILD_DIR/bin/llama-batched-bench" \
  -m "$MODEL" \
  -c 8192 \
  -b 4096 \
  -npp 128,512,1024 \
  -ntg 128 \
  -npl 1,2,4,8,16 \
  2>&1 | tee "$RESULTS_DIR/batched_bench.txt"

echo ""
echo "Results saved to $RESULTS_DIR/"
echo "  llama_bench.json   — t/s across prompt sizes, batch sizes, thread counts"
echo "  batched_bench.txt  — t/s vs parallel sequences (KV-cache/scheduling pressure)"
echo "  system_info.txt    — hardware context"
