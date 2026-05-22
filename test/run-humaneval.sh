#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
exec "${PYTHON:-python}" test/run_release_experiment.py --benchmark humaneval "$@"
