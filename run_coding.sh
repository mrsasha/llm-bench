#!/usr/bin/env bash
# Coding-focused run for a single machine: per model, context/speed, then HumanEval+/MBPP+.
#
#   ./run_coding.sh [results-dir] [model ...]
#
# Defaults: results/m4, and the models listed below. Models that aren't pulled
# yet are waited for, so this can start while `ollama pull` is still running.
# Both stages resume: ctx_bench skips models whose ctx-*.json exists,
# code_bench skips problem ids already in its JSONL.
set -u
cd "$(dirname "$0")"

N_PER_SET=25  # problems from HumanEval+ and from MBPP+ each
PY=${PYTHON:-python3}

out=${1:-results/m4}
shift || true
models=("$@")
if (( ${#models[@]} == 0 )); then
  models=(qwen3.5:9b bench-empero9b-q4 gemma4:12b qwen3.5:4b rnj-1:8b qwen2.5-coder:14b qwen2.5-coder:7b)
fi

curl -sf --max-time 5 http://localhost:11434/api/version > /dev/null \
  || { echo "Ollama is not running at localhost:11434" >&2; exit 1; }
mkdir -p "$out"

slug() { echo "$1" | tr ':/' '--'; }

for m in "${models[@]}"; do
  until ollama list | awk 'NR > 1 {print $1}' | grep -qx -e "$m" -e "$m:latest"; do
    echo "=== $(date +%H:%M) waiting for $m to be pulled"; sleep 60
  done
  f="$out/ctx-$(slug "$m").json"
  if [[ -f "$f" ]]; then
    echo "=== skip ctx $m"
  else
    echo "=== $(date +%H:%M) ctx $m"
    "$PY" ctx_bench.py --model "$m" --out "$f"
  fi
  echo "=== $(date +%H:%M) code $m"
  "$PY" code_bench.py --model "$m" --out "$out/code-$(slug "$m").jsonl" \
    --n-humaneval $N_PER_SET --n-mbpp $N_PER_SET
done
echo "=== $(date +%H:%M) done"
