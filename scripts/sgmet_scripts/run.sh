#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_ROOT/lib/common.sh"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/sgmet_scripts/run.sh --list
  bash scripts/sgmet_scripts/run.sh <config-name-or-path>

Examples:
  bash scripts/sgmet_scripts/run.sh supervised/e2e_scratch_semantic_2cls
  DRY_RUN=1 bash scripts/sgmet_scripts/run.sh supervised/e2e_scratch_semantic_2cls
  BACKGROUND=1 bash scripts/sgmet_scripts/run.sh supervised/e2e_scratch_semantic_2cls
  FOLDS=0 MAX_TRAIN_BATCHES=2 MAX_VAL_BATCHES=2 \
    bash scripts/sgmet_scripts/run.sh supervised/e2e_scratch_semantic_2cls

Environment controls:
  BACKGROUND=1       launch with nohup; use caffeinate on macOS when available
  DRY_RUN=1          print commands without executing them
  DATE_TAG=YYYYMMDD  override output date prefix
  FOLDS=0,1          override the folds in a config
USAGE
}

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

if [[ "$1" == "--list" ]]; then
  find "$SCRIPT_ROOT/configs" -type f -name '*.sh' \
    | sed "s#^${SCRIPT_ROOT}/configs/##; s#\.sh\$##" \
    | sort
  exit 0
fi

CONFIG_PATH="$(resolve_config_path "$1")"
# shellcheck disable=SC1090
source "$CONFIG_PATH"

: "${RUNNER:?Config must set RUNNER}"
: "${EXPERIMENT_NAME:?Config must set EXPERIMENT_NAME}"
: "${DATE_TAG:=$(date +%Y%m%d)}"

RUNNER_PATH="$SCRIPT_ROOT/runners/${RUNNER}.sh"
require_file "$RUNNER_PATH"

if bool_is_true "${BACKGROUND:-0}"; then
  mkdir -p "$REPO_ROOT/logs"
  LOG_FILE="${LOG_FILE:-$REPO_ROOT/logs/${DATE_TAG}_${EXPERIMENT_NAME}.out}"
  PID_FILE="${PID_FILE:-$REPO_ROOT/logs/${DATE_TAG}_${EXPERIMENT_NAME}.pid}"

  command=(bash "$RUNNER_PATH" "$CONFIG_PATH")
  if bool_is_true "${USE_CAFFEINATE:-1}" && command -v caffeinate >/dev/null 2>&1; then
    command=(caffeinate -dimsu "${command[@]}")
  fi

  nohup "${command[@]}" > "$LOG_FILE" 2>&1 < /dev/null &
  pid=$!
  echo "$pid" > "$PID_FILE"
  disown "$pid" 2>/dev/null || true

  echo "Started: $EXPERIMENT_NAME"
  echo "PID:     $pid"
  echo "Log:     $LOG_FILE"
  echo "PID file:$PID_FILE"
  echo "Follow:  tail -f '$LOG_FILE'"
else
  exec bash "$RUNNER_PATH" "$CONFIG_PATH"
fi
