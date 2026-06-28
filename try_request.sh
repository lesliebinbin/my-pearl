#!/usr/bin/env bash
set -euo pipefail

curl http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "pearl-ai/Llama-3.1-8B-Instruct-pearl",
    "messages": [
      {"role": "user", "content": "Explain Pearl mining in one sentence."}
    ],
    "max_tokens": 64,
    "temperature": 0.2
  }'
