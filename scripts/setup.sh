#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1; }
case "${1:-sync}" in
  lock)
    uv lock --project .
    uv lock --project environments/swebench --python 3.11
    uv lock --project environments/tbv2 --python 3.12
    echo 'Review and commit all three uv.lock files before checkpoint comparisons.'
    ;;
  sync)
    for project in . environments/swebench environments/tbv2; do
      test -f "$project/uv.lock" || { echo "Missing $project/uv.lock; run bash scripts/setup.sh lock on a connected machine." >&2; exit 1; }
      uv sync --locked --project "$project"
    done
    ;;
  *) echo 'Usage: bash scripts/setup.sh [lock|sync]' >&2; exit 1 ;;
esac
