#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p "$(dirname "${OUTPUT_FILE:-results-math/public_train_math.json}")"
mkdir -p "$(dirname "${LOG_FILE:-results-math/public_train_math.log}")"

exec "${PYTHON:-python}" -u experiments/math/run_math.py \
  --in_file "${INPUT_FILE:-project_datasets/MATH/processed_math/train.jsonl}" \
  --out_file "${OUTPUT_FILE:-results-math/public_train_math.json}" \
  --log_file "${LOG_FILE:-results-math/public_train_math.log}" \
  --selector_url "${SELECTOR_URL:?Set SELECTOR_URL}" \
  --selector_model "${SELECTOR_MODEL:?Set SELECTOR_MODEL}" \
  --selector_key "${SELECTOR_KEY:-EMPTY}" \
  --reasoning_url "${REASONING_URL:?Set REASONING_URL}" \
  --reasoning_model "${REASONING_MODEL:?Set REASONING_MODEL}" \
  --reasoning_key "${REASONING_KEY:-EMPTY}" \
  --supervisor_url "${SUPERVISOR_URL:?Set SUPERVISOR_URL}" \
  --supervisor_model "${SUPERVISOR_MODEL:?Set SUPERVISOR_MODEL}" \
  --supervisor_key "${SUPERVISOR_KEY:-EMPTY}" \
  --embedding_url "${EMBEDDING_URL:?Set EMBEDDING_URL}" \
  --embedding_model "${EMBEDDING_MODEL:?Set EMBEDDING_MODEL}" \
  --embedding_key "${EMBEDDING_KEY:-EMPTY}" \
  --max_turns "${MAX_TURNS:-7}" \
  --limit "${LIMIT:-2000}"
