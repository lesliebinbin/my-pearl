#!/usr/bin/env bash
set -euo pipefail

# Requires pearld and pearl-gateway to already be running.
# Mining work is triggered by inference requests with sufficiently large GEMMs.
export MINER_NO_MINING="${MINER_NO_MINING:-false}"
export MINER_SKIP_BLOCK_SUBMISSION="${MINER_SKIP_BLOCK_SUBMISSION:-false}"
export MINER_NO_GATEWAY="${MINER_NO_GATEWAY:-false}"
export PEARL_LOG_LEVEL="${PEARL_LOG_LEVEL:-INFO}"

uv run vllm serve pearl-ai/Llama-3.1-8B-Instruct-pearl \
  --port "${VLLM_PORT:-8002}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-8192}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.9}" \
  --enforce-eager
