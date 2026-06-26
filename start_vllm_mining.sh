#!/usr/bin/env zsh
VLLM_LOGGING_LEVEL=DEBUG uv run vllm serve pearl-ai/Gemma-4-31B-it-pearl \
  --port 8002 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 8192 \
  --enforce-eager

