#!/usr/bin/env bash
# Beast3 GPU bench — 4× RTX 3090, Ampere sm_86, PCIe, CUDA
# Usage: ./run_bench_beast3.sh /path/to/model.gguf [baseline|candidate]
#
# Run twice: once on the unmodified build (baseline), once after the mmvq.cu change.
# Compare llama_bench_baseline.json vs llama_bench_candidate.json for TG delta.

set -e

MODEL=${1:?Usage: $0 /path/to/model.gguf [baseline|candidate]}
TAG=${2:-candidate}

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$REPO_DIR/build_beast3"
RESULTS_DIR="$REPO_DIR/results/beast3"
mkdir -p "$RESULTS_DIR"

echo "=== [1/4] Building (CUDA sm_86, 4-GPU) ==="
cmake -B "$BUILD_DIR" "$REPO_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DLLAMA_PERF=OFF \
  -DGGML_PERF=OFF
cmake --build "$BUILD_DIR" --config Release -j"$(nproc)" \
  --target llama-bench llama-server

echo "=== [2/4] System info ==="
{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "tag: $TAG"
  echo "model: $MODEL"
  nvidia-smi --query-gpu=name,driver_version,memory.total,pcie.link.gen.current \
             --format=csv,noheader
  uname -a
} > "$RESULTS_DIR/system_info_${TAG}.txt"

echo "=== [3/4] llama-bench — TG and PP across batch sizes ==="
# ncols_dst=1 is the decode (TG) path — this is what the nwarps change targets.
# -p 512 = prefill tokens (PP measurement)
# -n 128 = generation tokens (TG measurement)
# -b 1   = batch size 1 = single-stream decode, ncols_dst=1 in the MMVQ kernel
# -ngl 99 = all layers on GPU
# --split-mode row = optimal for PCIe 4-GPU (reduces inter-GPU traffic vs layer)
"$BUILD_DIR/bin/llama-bench" \
  -m "$MODEL" \
  -p 512,1024,2048,4096 \
  -n 128,256 \
  -b 1,4,8 \
  -ngl 99 \
  --split-mode row \
  --flash-attn \
  -r 3 \
  -o json \
  > "$RESULTS_DIR/llama_bench_${TAG}.json"

echo "=== [4/4] TG-focused single-token decode sweep ==="
# This isolates the nwarps change: single-token decode (ncols_dst=1), warm KV cache.
# Run after prefill so KV cache is populated.
"$BUILD_DIR/bin/llama-bench" \
  -m "$MODEL" \
  -p 1024 \
  -n 512 \
  -b 1 \
  -ngl 99 \
  --split-mode row \
  --flash-attn \
  -r 5 \
  -o json \
  > "$RESULTS_DIR/llama_bench_tg_focused_${TAG}.json"

echo ""
echo "Results saved to $RESULTS_DIR/"
echo "  llama_bench_${TAG}.json            — PP and TG across batch/prompt sizes"
echo "  llama_bench_tg_focused_${TAG}.json — TG at n=512 for nwarps regression check"
echo ""
echo "To compare baseline vs candidate:"
echo "  python3 -c \""
echo "  import json, sys"
echo "  b = json.load(open('$RESULTS_DIR/llama_bench_baseline.json'))"
echo "  c = json.load(open('$RESULTS_DIR/llama_bench_candidate.json'))"
echo "  for brow, crow in zip(b, c):"
echo "    if brow.get('n_batch') == 1:"
echo "      tg_b = brow.get('tg_tokens_per_second', 0)"
echo "      tg_c = crow.get('tg_tokens_per_second', 0)"
echo "      pp_b = brow.get('pp_tokens_per_second', 0)"
echo "      pp_c = crow.get('pp_tokens_per_second', 0)"
echo "      print(f'TG: {tg_b:.1f} → {tg_c:.1f} t/s ({(tg_c/tg_b-1)*100:+.1f}%)')"
echo "      print(f'PP: {pp_b:.1f} → {pp_c:.1f} t/s ({(pp_c/pp_b-1)*100:+.1f}%)')"
echo "  \""
