#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_ROOT/lib/common.sh"

[[ $# -eq 1 ]] || fail "Usage: supervised_hpo_one_fold.sh <config.sh>"
CONFIG_PATH="$(resolve_config_path "$1")"
# shellcheck disable=SC1090
source "$CONFIG_PATH"

require_file "src/clinical_cluster_experts/train_supervised_downstream.py"
activate_conda_env
print_runtime

: "${EXPERIMENT_NAME:?}"
: "${CLUSTER_CSV_NAME:?}"
: "${NUM_SUMMARY_TOKENS:?}"

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
FOLD="${FOLD:-0}"
BASE_TOKEN_DIR="${BASE_TOKEN_DIR:-data/processed/tokenized_nhanes_v4}"
TOKEN_DIR="${TOKEN_DIR:-${BASE_TOKEN_DIR}/cv_splits/fold${FOLD}}"
CLUSTER_CSV="${TOKEN_DIR}/${CLUSTER_CSV_NAME}"
OUT_DIR="${OUT_DIR:-outputs/${DATE_TAG}_${EXPERIMENT_NAME}}"
TB_DIR="${TB_DIR:-runs/${DATE_TAG}_${EXPERIMENT_NAME}}"
STUDY_NAME="${STUDY_NAME:-${EXPERIMENT_NAME}}"
STORAGE="${STORAGE:-sqlite:///${OUT_DIR}/optuna_supervised.db}"

NUM_CLUSTERS="${NUM_CLUSTERS:-8}"
ACTIVE_CLUSTERS="${ACTIVE_CLUSTERS:-0,1,2,3,4,5,6}"
IGNORE_CLUSTERS="${IGNORE_CLUSTERS:-7}"
EPOCHS="${EPOCHS:-80}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-15}"
BATCH_SIZE="${BATCH_SIZE:-128}"
D_MODEL="${D_MODEL:-64}"
EXPERT_N_LAYERS="${EXPERT_N_LAYERS:-2}"
EXPERT_N_HEADS="${EXPERT_N_HEADS:-4}"
FUSION_N_LAYERS="${FUSION_N_LAYERS:-2}"
FUSION_N_HEADS="${FUSION_N_HEADS:-4}"
DROPOUT="${DROPOUT:-0.1}"
FEATURE_DROPOUT_PROB_MAX="${FEATURE_DROPOUT_PROB_MAX:-0.10}"
CLUSTER_DROPOUT_PROB_MAX="${CLUSTER_DROPOUT_PROB_MAX:-0.90}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-mps}"
LR_LOW="${LR_LOW:-2e-4}"
LR_HIGH="${LR_HIGH:-2e-3}"
WD_LOW="${WD_LOW:-1e-6}"
WD_HIGH="${WD_HIGH:-3e-3}"
OPTUNA_TRIALS="${OPTUNA_TRIALS:-14}"
OPTUNA_STARTUP_TRIALS="${OPTUNA_STARTUP_TRIALS:-8}"
OPTUNA_PRUNER_STARTUP_TRIALS="${OPTUNA_PRUNER_STARTUP_TRIALS:-8}"
OPTUNA_PRUNER_WARMUP_EPOCHS="${OPTUNA_PRUNER_WARMUP_EPOCHS:-15}"
USE_CLUSTER_EMBEDDING="${USE_CLUSTER_EMBEDDING:-0}"
EXPERT_INIT="${EXPERT_INIT:-pretrained}"
TRAIN_EXPERTS="${TRAIN_EXPERTS:-0}"
SUMMARY_DIAGNOSTICS="${SUMMARY_DIAGNOSTICS:-0}"
SUMMARY_AUX_LOSS="${SUMMARY_AUX_LOSS:-none}"
SUMMARY_AUX_WEIGHT="${SUMMARY_AUX_WEIGHT:-0.0}"
SUMMARY_AUX_WARMUP_EPOCHS="${SUMMARY_AUX_WARMUP_EPOCHS:-5}"
SUMMARY_ATTENTION_MARGIN="${SUMMARY_ATTENTION_MARGIN:-0.90}"
SUMMARY_OUTPUT_MARGIN="${SUMMARY_OUTPUT_MARGIN:-0.90}"
SUMMARY_MI_BETA="${SUMMARY_MI_BETA:-1.0}"
SUMMARY_MI_TEMPERATURE="${SUMMARY_MI_TEMPERATURE:-1.0}"
CLUSTER_DROPOUT_MAX_LOW="${CLUSTER_DROPOUT_MAX_LOW:-0.0}"
CLUSTER_DROPOUT_MAX_HIGH="${CLUSTER_DROPOUT_MAX_HIGH:-0.9}"

require_file "$CLUSTER_CSV"
mkdir -p "$OUT_DIR" "$TB_DIR"
save_run_manifest "$OUT_DIR" "$CONFIG_PATH"

optional_args=()
if [[ "$EXPERT_INIT" == "pretrained" ]]; then
  : "${PRETRAIN_ROOT:?PRETRAIN_ROOT is required when EXPERT_INIT=pretrained}"
  active_ids=()
  IFS=',' read -r -a active_ids <<< "$ACTIVE_CLUSTERS"
  for cluster_id in "${active_ids[@]}"; do
    checkpoint="${PRETRAIN_ROOT}/fold${FOLD}/cluster${cluster_id}/cluster_${cluster_id}_best.pt"
    require_file "$checkpoint"
    optional_args+=(--branch-checkpoint "${cluster_id}=${checkpoint}")
  done
  optional_args+=(--strict-branch-load)
else
  optional_args+=(--allow-random-active-branches)
fi
if [[ "$TRAIN_EXPERTS" == "1" ]]; then
  optional_args+=(--train-experts)
fi
if [[ "$USE_CLUSTER_EMBEDDING" != "1" ]]; then
  optional_args+=(--no-cluster-embedding)
fi
if [[ "$SUMMARY_DIAGNOSTICS" == "1" ]]; then
  optional_args+=(--summary-diagnostics)
fi
if [[ "${MAX_TRAIN_BATCHES:-0}" != "0" ]]; then
  optional_args+=(--max-train-batches "$MAX_TRAIN_BATCHES")
fi
if [[ "${MAX_VAL_BATCHES:-0}" != "0" ]]; then
  optional_args+=(--max-val-batches "$MAX_VAL_BATCHES")
fi

echo "Running one-fold HPO on fold $FOLD; test data are not used."
command=(
  python -m src.clinical_cluster_experts.train_supervised_downstream
  --token-dir "$TOKEN_DIR"
  --cluster-csv "$CLUSTER_CSV"
  --num-clusters "$NUM_CLUSTERS"
  --active-clusters "$ACTIVE_CLUSTERS"
  --ignore-clusters "$IGNORE_CLUSTERS"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --d-model "$D_MODEL"
  --num-summary-tokens "$NUM_SUMMARY_TOKENS"
  --expert-n-layers "$EXPERT_N_LAYERS"
  --expert-n-heads "$EXPERT_N_HEADS"
  --fusion-n-layers "$FUSION_N_LAYERS"
  --fusion-n-heads "$FUSION_N_HEADS"
  --dropout "$DROPOUT"
  --lr-schedule cosine
  --warmup-frac 0.05
  --min-lr-ratio 0.05
  --binary-loss focal
  --focal-gamma 1.0
  --feature-dropout-prob-max "$FEATURE_DROPOUT_PROB_MAX"
  --cluster-dropout-prob-max "$CLUSTER_DROPOUT_PROB_MAX"
  --summary-aux-loss "$SUMMARY_AUX_LOSS"
  --summary-aux-weight "$SUMMARY_AUX_WEIGHT"
  --summary-aux-warmup-epochs "$SUMMARY_AUX_WARMUP_EPOCHS"
  --summary-attention-margin "$SUMMARY_ATTENTION_MARGIN"
  --summary-output-margin "$SUMMARY_OUTPUT_MARGIN"
  --summary-mi-beta "$SUMMARY_MI_BETA"
  --summary-mi-temperature "$SUMMARY_MI_TEMPERATURE"
  --selection-metric macro_auroc
  --hpo-objective macro_auroc
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
  --seed "$SEED"
  --device "$DEVICE"
  --hpo
  --optuna-trials "$OPTUNA_TRIALS"
  --optuna-lr-low "$LR_LOW"
  --optuna-lr-high "$LR_HIGH"
  --optuna-weight-decay-low "$WD_LOW"
  --optuna-weight-decay-high "$WD_HIGH"
  --optuna-cluster-dropout-prob-max-low "$CLUSTER_DROPOUT_MAX_LOW"
  --optuna-cluster-dropout-prob-max-high "$CLUSTER_DROPOUT_MAX_HIGH"
  --optuna-startup-trials "$OPTUNA_STARTUP_TRIALS"
  --optuna-pruner median
  --optuna-pruner-startup-trials "$OPTUNA_PRUNER_STARTUP_TRIALS"
  --optuna-pruner-warmup-steps "$OPTUNA_PRUNER_WARMUP_EPOCHS"
  --optuna-sampler-seed "$SEED"
  --optuna-study-name "$STUDY_NAME"
  --optuna-storage "$STORAGE"
  --output-dir "$OUT_DIR"
  --log-dir "$TB_DIR"
  "${optional_args[@]}"
)
run_command "${command[@]}"
