#!/bin/zsh
# Daily live run: grade finished games, then publish today's predictions.
# Run once a day after the previous night's games end and before today's tip-offs
# (e.g. midday Singapore time = around midnight US Eastern).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
{
  echo "=== $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
  git pull --ff-only --quiet
  .venv/bin/python -m src.live score --commit
  .venv/bin/python -m src.live predict --commit
} >> logs/daily.log 2>&1
