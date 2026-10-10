#!/usr/bin/env bash
# Sequential quality runs. Ollama can't run qwen35-architecture requests in parallel.
#
#   ./run_all.sh               resume: skip stages whose results/quality-*.json exists
#   ./run_all.sh --fresh       delete quality results, logs and request cache, then run all
#   ./run_all.sh --fresh --yes same, without the confirmation prompt
set -u
cd "$(dirname "$0")"

OLLAMA_URL=http://localhost:11434
OLLAMA_WAIT_SECS=600

fresh=false
yes=false
for arg in "$@"; do
  case "$arg" in
    --fresh) fresh=true ;;
    --yes|-y) yes=true ;;
    -h|--help) sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg (see --help)" >&2; exit 2 ;;
  esac
done

ollama_up() { curl -sf --max-time 5 "$OLLAMA_URL/api/version" > /dev/null; }

# Ollama sometimes restarts mid-run (updates, memory pressure). Wait for it
# rather than letting every remaining stage fail with "connection refused".
wait_for_ollama() {
  ollama_up && return 0
  echo "WARNING: Ollama is not responding at $OLLAMA_URL. Waiting up to $((OLLAMA_WAIT_SECS / 60)) min..."
  local waited=0
  while ! ollama_up; do
    if (( waited >= OLLAMA_WAIT_SECS )); then
      echo "ERROR: Ollama still down. Start it, then rerun ./run_all.sh to resume." >&2
      exit 1
    fi
    sleep 15; waited=$((waited + 15))
  done
  echo "Ollama is back."
}

if ! ollama_up; then
  echo "WARNING: Ollama is not running at $OLLAMA_URL. Start the Ollama app (or 'ollama serve') and try again." >&2
  exit 1
fi

if $fresh; then
  shopt -s nullglob
  old=(results/quality-*.json results/log-*.txt results/mem-quality.csv)
  if (( ${#old[@]} )) || [[ -d cache ]]; then
    echo "--fresh will delete ${#old[@]} result/log file(s) and the request cache in cache/."
    if ! $yes; then
      read -r -p "Continue? [y/N] " answer
      [[ "$answer" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }
    fi
    rm -f "${old[@]}"
    rm -rf cache
  fi
  shopt -u nullglob
fi

# Memory samples for the whole run (VRAM, system RAM, Ollama's RSS); see mem_monitor.py.
# Git Bash's $$ isn't a Windows pid; /proc/$$/winpid is.
python mem_monitor.py --out results/mem-quality.csv --watch-pid "$(cat /proc/$$/winpid 2>/dev/null || echo $$)" &
MON_PID=$!
trap 'kill $MON_PID 2>/dev/null' EXIT

MMLU27=mmlu_flan_cot_zeroshot_abstract_algebra,mmlu_flan_cot_zeroshot_clinical_knowledge,mmlu_flan_cot_zeroshot_college_mathematics,mmlu_flan_cot_zeroshot_conceptual_physics,mmlu_flan_cot_zeroshot_formal_logic,mmlu_flan_cot_zeroshot_high_school_computer_science,mmlu_flan_cot_zeroshot_high_school_macroeconomics,mmlu_flan_cot_zeroshot_high_school_psychology,mmlu_flan_cot_zeroshot_human_aging,mmlu_flan_cot_zeroshot_logical_fallacies,mmlu_flan_cot_zeroshot_medical_genetics,mmlu_flan_cot_zeroshot_nutrition,mmlu_flan_cot_zeroshot_professional_law,mmlu_flan_cot_zeroshot_security_studies,mmlu_flan_cot_zeroshot_world_religions

# Stages whose result file already exists are skipped; delete the file to rerun one.
run() {
  local out="results/quality-$1-$2.json"
  if [[ -f "$out" ]]; then echo "=== skip $1 $2 ($out exists)"; return; fi
  wait_for_ollama
  echo "=== $(date +%H:%M) $1 $2"
  python run_eval.py "${@:3}" --model "$1" --out "$out" > "results/log-$1-$2.txt" 2>&1
  echo "exit=$?"
}

for m in bench-empero9b-q4 bench-qwen35-9b; do
  run $m full --tasks gsm8k_cot,ifeval --limit 50
  run $m full-mmlu --tasks mmlu_flan_cot_zeroshot --limit 1
done
run bench-gemma4-12b full --tasks gsm8k_cot,ifeval --limit 50 --temperature 1.0
run bench-gemma4-12b full-mmlu --tasks mmlu_flan_cot_zeroshot --limit 1 --temperature 1.0
run bench-qwen38-27b sub --tasks gsm8k_cot,ifeval --limit 15
run bench-qwen38-27b sub-mmlu --tasks $MMLU27 --limit 1
echo "=== $(date +%H:%M) done"
