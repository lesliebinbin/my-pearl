#!/usr/bin/env zsh
VLLM_LOGGING_LEVEL=DEBUG uv run vllm serve pearl-ai/Llama-3.1-8B-Instruct-pearl \
  --port 8002 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 4096 \
  --enforce-eager

