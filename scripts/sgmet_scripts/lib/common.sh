#!/usr/bin/env bash
# Shared helpers for all SGMET shell runners.
# This file is sourced; do not execute it directly.

SGMET_SCRIPTS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Prefer Git because it remains correct if the scripts are moved into a deeper
# subfolder. Fall back to the known layout: <repo>/scripts/sgmet_scripts/lib.
if command -v git >/dev/null 2>&1; then
  REPO_ROOT="$(git -C "$SGMET_SCRIPTS_ROOT" rev-parse --show-toplevel 2>/dev/null || true)"
else
  REPO_ROOT=""
fi
if [[ -z "$REPO_ROOT" ]]; then
  REPO_ROOT="$(cd -- "$SGMET_SCRIPTS_ROOT/../.." && pwd)"
fi

cd "$REPO_ROOT"

fail() {
  echo "[ERROR] $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || fail "Missing file: $1"
}

require_dir() {
  [[ -d "$1" ]] || fail "Missing directory: $1"
}

resolve_config_path() {
  local requested="$1"
  if [[ -f "$requested" ]]; then
    (cd -- "$(dirname -- "$requested")" && printf '%s/%s\n' "$PWD" "$(basename -- "$requested")")
    return
  fi
  if [[ -f "$SGMET_SCRIPTS_ROOT/$requested" ]]; then
    printf '%s/%s\n' "$SGMET_SCRIPTS_ROOT" "$requested"
    return
  fi
  if [[ -f "$SGMET_SCRIPTS_ROOT/configs/$requested" ]]; then
    printf '%s/%s\n' "$SGMET_SCRIPTS_ROOT/configs" "$requested"
    return
  fi
  if [[ -f "$SGMET_SCRIPTS_ROOT/configs/${requested}.sh" ]]; then
    printf '%s/%s.sh\n' "$SGMET_SCRIPTS_ROOT/configs" "$requested"
    return
  fi
  fail "Could not resolve config: $requested"
}

activate_conda_env() {
  local env_name="${CONDA_ENV:-di-lab}"
  local conda_sh=""
  for candidate in \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    if [[ -f "$candidate" ]]; then
      conda_sh="$candidate"
      break
    fi
  done
  [[ -n "$conda_sh" ]] || fail "Could not find conda.sh."
  # shellcheck disable=SC1090
  source "$conda_sh"
  conda activate "$env_name"
}

print_runtime() {
  echo "Repository root: $REPO_ROOT"
  echo "Conda environment: ${CONDA_DEFAULT_ENV:-unknown}"
  echo "Python: $(command -v python)"
  python - <<'PY'
import sys
import numpy as np
import torch
print("Python executable:", sys.executable)
print("NumPy:", np.__version__)
print("Torch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
PY
}

run_command() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '[DRY RUN]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

save_run_manifest() {
  local output_root="$1"
  local config_path="$2"
  mkdir -p "$output_root"

  cp "$config_path" "$output_root/resolved_experiment_config.sh"
  {
    echo "timestamp=$(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo "repo_root=$REPO_ROOT"
    echo "config=$config_path"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "git_status_begin"
    git status --short 2>/dev/null || true
    echo "git_status_end"
  } > "$output_root/run_manifest.txt"
}

bool_is_true() {
  [[ "${1:-0}" == "1" || "${1:-}" == "true" || "${1:-}" == "yes" ]]
}
