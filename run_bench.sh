#!/usr/bin/env bash
# Usage: ./run_bench.sh /path/to/model.gguf
# Builds llama.cpp, runs llama-bench + batched-bench, saves JSON results.

set -e

MODEL=${1:?Usage: $0 /path/to/model.gguf}
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$REPO_DIR/build"
RESULTS_DIR="$REPO_DIR/results"
mkdir -p "$RESULTS_DIR"

echo "=== [0/4] Committing script before build (bakes new hash into binary) ==="
git -C "$REPO_DIR" add run_bench.sh
git -C "$REPO_DIR" commit -m "perf: enable -march=native -O3 for Xeon 8581C optimisation" \
  --allow-empty || true

echo "=== [1/4] Building ==="
cmake -B "$BUILD_DIR" "$REPO_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_PERF=ON \
  -DGGML_PERF=ON \
  -DCMAKE_CXX_FLAGS='-march=native -O3' \
  -DCMAKE_C_FLAGS='-march=native -O3'
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
export OMP_WAIT_POLICY=passive
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Pass 1: KV-cache quantisation sweep (flash attention off by default).
# -fa 1 is incompatible with quantised KV types, so this pass keeps the
# q8_0 variants and omits the -fa flag entirely.
echo "--- [3a] KV-quant sweep (ctk/ctv: f16,q8_0; fa=0) ---"
numactl --localalloc \
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

# Pass 2: Flash-attention sweep (fa 0 vs 1) with f16 KV only, since
# fa=1 is incompatible with quantised KV cache types.
echo "--- [3b] Flash-attention sweep (ctk/ctv: f16; fa=0,1) ---"
numactl --localalloc \
  "$BUILD_DIR/bin/llama-bench" \
  -m "$MODEL" \
  -p 128,512,1024,2048 \
  -n 128 \
  -b 512,2048 \
  -ub 128,256,512,1024 \
  -t 1,8,16,32,"$(nproc)" \
  -ctk f16 \
  -ctv f16 \
  -fa 0,1 \
  -r 5 \
  -o json -oe sql \
  >> "$RESULTS_DIR/llama_bench.json" \
  2> >(sqlite3 "$RESULTS_DIR/llama_bench.sqlite")

unset OMP_WAIT_POLICY OMP_PROC_BIND OMP_PLACES

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
