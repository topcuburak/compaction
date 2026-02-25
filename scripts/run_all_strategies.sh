#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-data/sample_browsecomp.jsonl}"
COMMON_ARGS=(
  --dataset "$DATASET"
  --max-questions "${MAX_QUESTIONS:-10}"
  --max-steps "${MAX_STEPS:-24}"
  --model "${MODEL_NAME:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
  --base-url "${OPENAI_BASE_URL:-http://localhost:8000/v1}"
  --api-key "${OPENAI_API_KEY:-EMPTY}"
)

python run_browsecomp_langchain.py "${COMMON_ARGS[@]}" --strategy none
python run_browsecomp_langchain.py "${COMMON_ARGS[@]}" --strategy trim
python run_browsecomp_langchain.py "${COMMON_ARGS[@]}" --strategy summarize
