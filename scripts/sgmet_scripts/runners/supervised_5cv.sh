#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_ROOT/lib/common.sh"

[[ $# -eq 1 ]] || fail "Usage: supervised_5cv.sh <config.sh>"
CONFIG_PATH="$(resolve_config_path "$1")"
# shellcheck disable=SC1090
source "$CONFIG_PATH"

require_file "src/clinical_cluster_experts/train_supervised_downstream.py"
require_file "src/clinical_cluster_experts/summary_auxiliary_losses.py"
activate_conda_env
print_runtime

: "${EXPERIMENT_NAME:?}"
: "${EXPERT_INIT:?Use 'random' or 'pretrained'}"
: "${TRAIN_EXPERTS:?Use 0 or 1}"
: "${CLUSTER_CSV_NAME:?}"
: "${NUM_SUMMARY_TOKENS:?}"

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
BASE_TOKEN_DIR="${BASE_TOKEN_DIR:-data/processed/tokenized_nhanes_v4}"
CV_DIR="${CV_DIR:-${BASE_TOKEN_DIR}/cv_splits}"
OUT_ROOT="${OUT_ROOT:-outputs/${DATE_TAG}_${EXPERIMENT_NAME}}"
TB_ROOT="${TB_ROOT:-runs/${DATE_TAG}_${EXPERIMENT_NAME}}"

NUM_CLUSTERS="${NUM_CLUSTERS:-8}"  # IDs 0..7; cluster 7 is deliberately ignored.
ACTIVE_CLUSTERS="${ACTIVE_CLUSTERS:-0,1,2,3,4,5,6}"
IGNORE_CLUSTERS="${IGNORE_CLUSTERS:-7}"
FOLDS="${FOLDS:-0,1,2,3,4}"

EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-128}"
D_MODEL="${D_MODEL:-64}"
EXPERT_N_LAYERS="${EXPERT_N_LAYERS:-2}"
EXPERT_N_HEADS="${EXPERT_N_HEADS:-4}"
FUSION_N_LAYERS="${FUSION_N_LAYERS:-2}"
FUSION_N_HEADS="${FUSION_N_HEADS:-4}"
DROPOUT="${DROPOUT:-0.1}"
LR="${LR:-8e-4}"
EXPERT_LR="${EXPERT_LR:-8e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
FEATURE_DROPOUT_PROB_MAX="${FEATURE_DROPOUT_PROB_MAX:-0.10}"
CLUSTER_DROPOUT_PROB_MAX="${CLUSTER_DROPOUT_PROB_MAX:-0.90}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-mps}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-0}"
USE_CLUSTER_EMBEDDING="${USE_CLUSTER_EMBEDDING:-0}"
SUMMARY_DIAGNOSTICS="${SUMMARY_DIAGNOSTICS:-0}"
SUMMARY_AUX_LOSS="${SUMMARY_AUX_LOSS:-none}"
SUMMARY_AUX_WEIGHT="${SUMMARY_AUX_WEIGHT:-0}"
SUMMARY_AUX_WARMUP_EPOCHS="${SUMMARY_AUX_WARMUP_EPOCHS:-5}"
SUMMARY_ATTENTION_MARGIN="${SUMMARY_ATTENTION_MARGIN:-0.90}"
SUMMARY_OUTPUT_MARGIN="${SUMMARY_OUTPUT_MARGIN:-0.90}"
SUMMARY_MI_BETA="${SUMMARY_MI_BETA:-1.0}"
SUMMARY_MI_TEMPERATURE="${SUMMARY_MI_TEMPERATURE:-1.0}"

if [[ "$EXPERT_INIT" == "pretrained" ]]; then
  : "${PRETRAIN_ROOT:?A pretrained experiment must set PRETRAIN_ROOT}"
elif [[ "$EXPERT_INIT" != "random" ]]; then
  fail "EXPERT_INIT must be 'random' or 'pretrained', got: $EXPERT_INIT"
fi

if [[ "$TRAIN_EXPERTS" != "0" && "$TRAIN_EXPERTS" != "1" ]]; then
  fail "TRAIN_EXPERTS must be 0 or 1."
fi
if [[ "$EXPERT_INIT" == "random" && "$TRAIN_EXPERTS" != "1" ]]; then
  fail "Randomly initialized experts must be trainable (TRAIN_EXPERTS=1)."
fi

mkdir -p "$OUT_ROOT" "$TB_ROOT"
save_run_manifest "$OUT_ROOT" "$CONFIG_PATH"

fold_ids=()
IFS=',' read -r -a fold_ids <<< "$FOLDS"

for fold in "${fold_ids[@]}"; do
  token_dir="${CV_DIR}/fold${fold}"
  cluster_csv="${token_dir}/${CLUSTER_CSV_NAME}"
  require_file "${token_dir}/train_tokens.pt"
  require_file "$cluster_csv"

  expert_init_args=()
  if [[ "$EXPERT_INIT" == "random" ]]; then
    expert_init_args+=(--allow-random-active-branches)
  else
    active_ids=()
    IFS=',' read -r -a active_ids <<< "$ACTIVE_CLUSTERS"
    for cluster_id in "${active_ids[@]}"; do
      checkpoint="${PRETRAIN_ROOT}/fold${fold}/cluster${cluster_id}/cluster_${cluster_id}_best.pt"
      require_file "$checkpoint"
      expert_init_args+=(--branch-checkpoint "${cluster_id}=${checkpoint}")
    done
    expert_init_args+=(--strict-branch-load)
  fi

  train_expert_args=()
  if [[ "$TRAIN_EXPERTS" == "1" ]]; then
    train_expert_args+=(--train-experts --expert-lr "$EXPERT_LR")
  fi

  optional_args=()
  if [[ "$USE_CLUSTER_EMBEDDING" != "1" ]]; then
    optional_args+=(--no-cluster-embedding)
  fi
  if [[ "$SUMMARY_DIAGNOSTICS" == "1" ]]; then
    optional_args+=(--summary-diagnostics)
  fi
  if [[ "$MAX_TRAIN_BATCHES" != "0" ]]; then
    optional_args+=(--max-train-batches "$MAX_TRAIN_BATCHES")
  fi
  if [[ "$MAX_VAL_BATCHES" != "0" ]]; then
    optional_args+=(--max-val-batches "$MAX_VAL_BATCHES")
  fi

  out_dir="${OUT_ROOT}/fold${fold}"
  tb_dir="${TB_ROOT}/fold${fold}"
  mkdir -p "$out_dir" "$tb_dir"

  echo
  echo "======================================================================"
  echo "Experiment:             $EXPERIMENT_NAME"
  echo "Fold:                   $fold"
  echo "Expert initialization:  $EXPERT_INIT"
  echo "Experts trainable:      $TRAIN_EXPERTS"
  echo "Cluster CSV:            $cluster_csv"
  echo "Summary tokens/expert:  $NUM_SUMMARY_TOKENS"
  echo "Summary aux loss:       $SUMMARY_AUX_LOSS"
  echo "Summary aux weight:     $SUMMARY_AUX_WEIGHT"
  echo "Output:                 $out_dir"
  echo "======================================================================"

  command=(
    python -m src.clinical_cluster_experts.train_supervised_downstream
    --token-dir "$token_dir"
    --cluster-csv "$cluster_csv"
    --num-clusters "$NUM_CLUSTERS"
    --active-clusters "$ACTIVE_CLUSTERS"
    --ignore-clusters "$IGNORE_CLUSTERS"
    "${expert_init_args[@]}"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --d-model "$D_MODEL"
    --num-summary-tokens "$NUM_SUMMARY_TOKENS"
    --summary-aux-loss "$SUMMARY_AUX_LOSS"
    --summary-aux-weight "$SUMMARY_AUX_WEIGHT"
    --summary-aux-warmup-epochs "$SUMMARY_AUX_WARMUP_EPOCHS"
    --summary-attention-margin "$SUMMARY_ATTENTION_MARGIN"
    --summary-output-margin "$SUMMARY_OUTPUT_MARGIN"
    --summary-mi-beta "$SUMMARY_MI_BETA"
    --summary-mi-temperature "$SUMMARY_MI_TEMPERATURE"
    --expert-n-layers "$EXPERT_N_LAYERS"
    --expert-n-heads "$EXPERT_N_HEADS"
    --fusion-n-layers "$FUSION_N_LAYERS"
    --fusion-n-heads "$FUSION_N_HEADS"
    --dropout "$DROPOUT"
    --lr "$LR"
    "${train_expert_args[@]}"
    --weight-decay "$WEIGHT_DECAY"
    --lr-schedule cosine
    --warmup-frac 0.05
    --min-lr-ratio 0.05
    --binary-loss focal
    --focal-gamma 1.0
    --feature-dropout-prob-max "$FEATURE_DROPOUT_PROB_MAX"
    --cluster-dropout-prob-max "$CLUSTER_DROPOUT_PROB_MAX"
    --selection-metric macro_auroc
    --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
    --seed "$SEED"
    --device "$DEVICE"
    --output-dir "$out_dir"
    --log-dir "$tb_dir"
    "${optional_args[@]}"
  )
  run_command "${command[@]}"
done

echo
echo "Completed supervised experiment: $EXPERIMENT_NAME"
echo "Outputs: $OUT_ROOT"
