#!/usr/bin/env bash
# Run the 8 TextToKids EMOTYC evaluation configurations.
# Dataset: TextToKids only. No CyberAggAdo commands are generated here.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PREDICT_SCRIPT="$ROOT_DIR/emotyc_predict.py"
GOLD_FILE="$ROOT_DIR/golds/emotexttokids_gold_flat.xlsx"
RESULTS_BASE="$ROOT_DIR/results/TextToKids"
BATCH_SIZE=490

CONFIG_NAMES=(
  "NoContextTemplateSansEspace"
  "NoContextTemplateSansEspaceMode05"
  "NoContextTemplateAvecEspace"
  "NoContextTemplateAvecEspaceMode05"
  "ContextTemplateSansEspace"
  "ContextTemplateSansEspaceMode05"
  "ContextTemplateAvecEspace"
  "ContextTemplateAvecEspaceMode05"
)

CONFIG_CONTEXT=(
  "0"
  "0"
  "0"
  "0"
  "1"
  "1"
  "1"
  "1"
)

CONFIG_TEMPLATE=(
  "bca"
  "bca"
  "bca_spaced"
  "bca_spaced"
  "bca"
  "bca"
  "bca_spaced"
  "bca_spaced"
)

CONFIG_MODE_THRESHOLD=(
  "0.06"
  "0.05"
  "0.06"
  "0.05"
  "0.06"
  "0.05"
  "0.06"
  "0.05"
)

if [[ ! -f "$PREDICT_SCRIPT" ]]; then
  echo "ERROR: missing predict script: $PREDICT_SCRIPT" >&2
  exit 1
fi

if [[ ! -f "$GOLD_FILE" ]]; then
  echo "ERROR: missing TextToKids gold file: $GOLD_FILE" >&2
  exit 1
fi

mkdir -p "$RESULTS_BASE"

printf '=%.0s' {1..80}; echo
printf 'EMOTYC TextToKids batch evaluation\n'
printf '  python       : %s\n' "$PYTHON_BIN"
printf '  script       : %s\n' "$PREDICT_SCRIPT"
printf '  gold         : %s\n' "$GOLD_FILE"
printf '  results base : %s\n' "$RESULTS_BASE"
printf '  batch size   : %s\n' "$BATCH_SIZE"
printf '=%.0s' {1..80}; echo

failed=0
start_all=$(date +%s)

for i in "${!CONFIG_NAMES[@]}"; do
  run_index=$((i + 1))
  name="${CONFIG_NAMES[$i]}"
  use_context="${CONFIG_CONTEXT[$i]}"
  template="${CONFIG_TEMPLATE[$i]}"
  mode_threshold="${CONFIG_MODE_THRESHOLD[$i]}"
  out_dir="$RESULTS_BASE/$name"
  log_file="$out_dir/run.log"

  mkdir -p "$out_dir"

  cmd=(
    "$PYTHON_BIN" "$PREDICT_SCRIPT"
    --xlsx "$GOLD_FILE"
    --out_dir "$out_dir"
    --template "$template"
    --batch-size "$BATCH_SIZE"
    --mode-threshold "$mode_threshold"
  )

  if [[ "$use_context" == "1" ]]; then
    cmd+=(--use-context)
  fi

  echo
  printf -- '-%.0s' {1..80}; echo
  printf '[%d/8] %s\n' "$run_index" "$name"
  printf '  template       : %s\n' "$template"
  printf '  mode-threshold : %s\n' "$mode_threshold"
  printf '  use-context    : %s\n' "$([[ "$use_context" == "1" ]] && echo yes || echo no)"
  printf '  out_dir        : %s\n' "$out_dir"
  printf '  log            : %s\n' "$log_file"
  printf '  command        :'; printf ' %q' "${cmd[@]}"; echo
  printf -- '-%.0s' {1..80}; echo

  set +e
  "${cmd[@]}" 2>&1 | tee "$log_file"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ "$rc" -ne 0 ]]; then
    echo "ERROR: run failed: $name rc=$rc" >&2
    failed=1
  else
    echo "OK: run completed: $name"
  fi
done

elapsed=$(( $(date +%s) - start_all ))
echo
printf '=%.0s' {1..80}; echo
printf 'TextToKids batch complete in %ss\n' "$elapsed"
printf 'Results base: %s\n' "$RESULTS_BASE"
printf '=%.0s' {1..80}; echo

if [[ "$failed" -ne 0 ]]; then
  echo "At least one run failed; inspect per-run run.log files under $RESULTS_BASE." >&2
  exit 1
fi
