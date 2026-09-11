#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_ROOT/lib/common.sh"

[[ $# -eq 1 ]] || fail "Usage: cluster_ablation_5cv.sh <config.sh>"
CONFIG_PATH="$(resolve_config_path "$1")"
# shellcheck disable=SC1090
source "$CONFIG_PATH"

require_file "src/clinical_cluster_experts/inference_5cv.py"
require_file "src/clinical_cluster_experts/aggregate_sgmet_cluster_ablation_results.py"
activate_conda_env
print_runtime

: "${EXPERIMENT_NAME:?}"
: "${CHECKPOINT_ROOT:?}"
: "${CLUSTER_CSV_NAME:?}"

CV_ROOT="${CV_ROOT:-data/processed/tokenized_nhanes_v4/cv_splits}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CHECKPOINT_ROOT}/cluster_ablation_test}"
SPLIT="${SPLIT:-test}"
FOLDS="${FOLDS:-0,1,2,3,4}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-best.pt}"
NUM_CLUSTERS="${NUM_CLUSTERS:-8}"
IGNORED_CLUSTERS="${IGNORED_CLUSTERS:-7}"
D_MODEL="${D_MODEL:-64}"
EXPERT_N_LAYERS="${EXPERT_N_LAYERS:-2}"
EXPERT_N_HEADS="${EXPERT_N_HEADS:-4}"
FUSION_N_LAYERS="${FUSION_N_LAYERS:-2}"
FUSION_N_HEADS="${FUSION_N_HEADS:-4}"
DROPOUT="${DROPOUT:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-mps}"
THRESHOLD="${THRESHOLD:-0.5}"
USE_CLUSTER_EMBEDDING="${USE_CLUSTER_EMBEDDING:-0}"
SAVE_SUMMARY_DIAGNOSTICS="${SAVE_SUMMARY_DIAGNOSTICS:-0}"
NUM_SUMMARY_TOKENS="${NUM_SUMMARY_TOKENS:-}"  # empty = read from checkpoint

mkdir -p "$OUTPUT_ROOT"
save_run_manifest "$OUTPUT_ROOT" "$CONFIG_PATH"

common_args=(
  --run-cv
  --cv-root "$CV_ROOT"
  --checkpoint-root "$CHECKPOINT_ROOT"
  --split "$SPLIT"
  --folds "$FOLDS"
  --checkpoint-name "$CHECKPOINT_NAME"
  --cluster-csv-name "$CLUSTER_CSV_NAME"
  --num-clusters "$NUM_CLUSTERS"
  --ignore-clusters "$IGNORED_CLUSTERS"
  --d-model "$D_MODEL"
  --expert-n-layers "$EXPERT_N_LAYERS"
  --expert-n-heads "$EXPERT_N_HEADS"
  --fusion-n-layers "$FUSION_N_LAYERS"
  --fusion-n-heads "$FUSION_N_HEADS"
  --dropout "$DROPOUT"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --device "$DEVICE"
  --threshold "$THRESHOLD"
  --save-cv-results
)
if [[ "$USE_CLUSTER_EMBEDDING" != "1" ]]; then
  common_args+=(--no-cluster-embedding)
fi
if [[ -n "$NUM_SUMMARY_TOKENS" ]]; then
  common_args+=(--num-summary-tokens "$NUM_SUMMARY_TOKENS")
fi
if [[ "$SAVE_SUMMARY_DIAGNOSTICS" == "1" ]]; then
  common_args+=(--save-summary-diagnostics)
fi

run_condition() {
  local condition_name="$1"
  local active_clusters="$2"
  local condition_dir="${OUTPUT_ROOT}/${condition_name}"

  echo
  echo "Condition: $condition_name | active clusters: $active_clusters"
  command=(
    python -m src.clinical_cluster_experts.inference_5cv
    "${common_args[@]}"
    --active-clusters "$active_clusters"
    --cv-output-dir "$condition_dir"
  )
  run_command "${command[@]}"
}

run_condition full "0,1,2,3,4,5,6"
run_condition without_cluster_0 "1,2,3,4,5,6"
run_condition without_cluster_1 "0,2,3,4,5,6"
run_condition without_cluster_2 "0,1,3,4,5,6"
run_condition without_cluster_3 "0,1,2,4,5,6"
run_condition without_cluster_4 "0,1,2,3,5,6"
run_condition without_cluster_5 "0,1,2,3,4,6"
run_condition without_cluster_6 "0,1,2,3,4,5"

run_command python -m src.clinical_cluster_experts.aggregate_sgmet_cluster_ablation_results \
  --input-root "$OUTPUT_ROOT" \
  --output-dir "$OUTPUT_ROOT/summary"

echo "Cluster ablation completed: $OUTPUT_ROOT/summary"
