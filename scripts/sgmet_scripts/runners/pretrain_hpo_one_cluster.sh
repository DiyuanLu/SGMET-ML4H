#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_ROOT/lib/common.sh"

[[ $# -eq 1 ]] || fail "Usage: pretrain_hpo_one_cluster.sh <config.sh>"
CONFIG_PATH="$(resolve_config_path "$1")"
# shellcheck disable=SC1090
source "$CONFIG_PATH"

require_file "src/clinical_cluster_experts/pretrain_expert_reconstruction.py"
activate_conda_env
print_runtime

: "${EXPERIMENT_NAME:?}"
: "${CLUSTER_CSV_NAME:?}"
: "${CLUSTER_ID:?}"
: "${NUM_SUMMARY_TOKENS:?}"

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
BASE_TOKEN_DIR="${BASE_TOKEN_DIR:-data/processed/tokenized_nhanes_v5}"
CV_DIR="${CV_DIR:-${BASE_TOKEN_DIR}/cv_splits}"
FOLD="${FOLD:-0}"
TOKEN_DIR="${CV_DIR}/fold${FOLD}"
CLUSTER_CSV="${TOKEN_DIR}/${CLUSTER_CSV_NAME}"

OUT_ROOT="${OUT_ROOT:-outputs/${DATE_TAG}_${EXPERIMENT_NAME}}"
TB_ROOT="${TB_ROOT:-runs/${DATE_TAG}_${EXPERIMENT_NAME}}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/fold${FOLD}/cluster${CLUSTER_ID}}"
TB_DIR="${TB_DIR:-${TB_ROOT}/fold${FOLD}/cluster${CLUSTER_ID}}"

NUM_CLUSTERS="${NUM_CLUSTERS:-7}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
D_MODEL="${D_MODEL:-64}"
N_LAYERS="${N_LAYERS:-2}"
N_HEADS="${N_HEADS:-4}"
DROPOUT="${DROPOUT:-0.1}"
FEATURE_MASK_PROB_MAX="${FEATURE_MASK_PROB_MAX:-0.25}"
SCHEMA_DROPOUT_PROB_MAX="${SCHEMA_DROPOUT_PROB_MAX:-0.10}"
CATEGORICAL_WEIGHT="${CATEGORICAL_WEIGHT:-1.0}"
CAT_LOSS="${CAT_LOSS:-focal}"
FOCAL_GAMMA="${FOCAL_GAMMA:-1.0}"
VAL_MASK_PASSES="${VAL_MASK_PASSES:-3}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
WARMUP_FRAC="${WARMUP_FRAC:-0.05}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.05}"
SEED="${SEED:-45}"
DEVICE="${DEVICE:-mps}"

SUMMARY_AUX_LOSS="${SUMMARY_AUX_LOSS:-none}"
SUMMARY_AUX_WEIGHT="${SUMMARY_AUX_WEIGHT:-0.0}"
SUMMARY_AUX_WARMUP_EPOCHS="${SUMMARY_AUX_WARMUP_EPOCHS:-5}"
SUMMARY_ATTENTION_MARGIN="${SUMMARY_ATTENTION_MARGIN:-0.9}"
SUMMARY_OUTPUT_MARGIN="${SUMMARY_OUTPUT_MARGIN:-0.9}"
SUMMARY_MI_BETA="${SUMMARY_MI_BETA:-1.0}"
SUMMARY_MI_TEMPERATURE="${SUMMARY_MI_TEMPERATURE:-1.0}"

OPTUNA_TRIALS="${OPTUNA_TRIALS:-30}"
OPTUNA_LR_LOW="${OPTUNA_LR_LOW:-1e-4}"
OPTUNA_LR_HIGH="${OPTUNA_LR_HIGH:-3e-3}"
OPTUNA_WEIGHT_DECAY_LOW="${OPTUNA_WEIGHT_DECAY_LOW:-1e-7}"
OPTUNA_WEIGHT_DECAY_HIGH="${OPTUNA_WEIGHT_DECAY_HIGH:-1e-3}"
OPTUNA_STARTUP_TRIALS="${OPTUNA_STARTUP_TRIALS:-8}"
OPTUNA_PRUNER_STARTUP_TRIALS="${OPTUNA_PRUNER_STARTUP_TRIALS:-5}"
OPTUNA_PRUNER_WARMUP_STEPS="${OPTUNA_PRUNER_WARMUP_STEPS:-10}"
OPTUNA_STUDY_NAME="${OPTUNA_STUDY_NAME:-${EXPERIMENT_NAME}}"
OPTUNA_STORAGE="${OPTUNA_STORAGE:-sqlite:///${OUT_DIR}/optuna_cluster_${CLUSTER_ID}.db}"

MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-0}"

require_file "${TOKEN_DIR}/train_tokens.pt"
require_file "${TOKEN_DIR}/val_tokens.pt"
require_file "$CLUSTER_CSV"

mkdir -p "$OUT_DIR" "$TB_DIR"
save_run_manifest "$OUT_ROOT" "$CONFIG_PATH"

echo
echo "======================================================================"
echo "Experiment:                    $EXPERIMENT_NAME"
echo "Fold / cluster:                $FOLD / $CLUSTER_ID"
echo "Token dir:                     $TOKEN_DIR"
echo "Cluster CSV:                   $CLUSTER_CSV"
echo "Summary tokens/expert:         $NUM_SUMMARY_TOKENS"
echo "Summary auxiliary loss:        $SUMMARY_AUX_LOSS"
echo "Summary auxiliary weight:      $SUMMARY_AUX_WEIGHT"
echo "MI beta / temperature:         $SUMMARY_MI_BETA / $SUMMARY_MI_TEMPERATURE"
echo "HPO trials:                    $OPTUNA_TRIALS"
echo "LR search:                     [$OPTUNA_LR_LOW, $OPTUNA_LR_HIGH]"
echo "Weight decay search:           [$OPTUNA_WEIGHT_DECAY_LOW, $OPTUNA_WEIGHT_DECAY_HIGH]"
echo "Epochs / complete trial:       $EPOCHS"
echo "Output:                        $OUT_DIR"
echo "TensorBoard:                   $TB_DIR"
echo "======================================================================"

command=(
  python -m src.clinical_cluster_experts.pretrain_expert_reconstruction
  --token-dir "$TOKEN_DIR"
  --cluster-csv "$CLUSTER_CSV"
  --num-clusters "$NUM_CLUSTERS"
  --cluster-id "$CLUSTER_ID"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --d-model "$D_MODEL"
  --num-summary-tokens "$NUM_SUMMARY_TOKENS"
  --n-layers "$N_LAYERS"
  --n-heads "$N_HEADS"
  --dropout "$DROPOUT"
  --feature-mask-prob-max "$FEATURE_MASK_PROB_MAX"
  --schema-dropout-prob-max "$SCHEMA_DROPOUT_PROB_MAX"
  --categorical-weight "$CATEGORICAL_WEIGHT"
  --cat-loss "$CAT_LOSS"
  --focal-gamma "$FOCAL_GAMMA"
  --val-mask-passes "$VAL_MASK_PASSES"
  --lr-schedule "$LR_SCHEDULE"
  --warmup-frac "$WARMUP_FRAC"
  --min-lr-ratio "$MIN_LR_RATIO"
  --summary-aux-loss "$SUMMARY_AUX_LOSS"
  --summary-aux-weight "$SUMMARY_AUX_WEIGHT"
  --summary-aux-warmup-epochs "$SUMMARY_AUX_WARMUP_EPOCHS"
  --summary-attention-margin "$SUMMARY_ATTENTION_MARGIN"
  --summary-output-margin "$SUMMARY_OUTPUT_MARGIN"
  --summary-mi-beta "$SUMMARY_MI_BETA"
  --summary-mi-temperature "$SUMMARY_MI_TEMPERATURE"
  --seed "$SEED"
  --device "$DEVICE"
  --output-dir "$OUT_DIR"
  --log-dir "$TB_DIR"
  --hpo
  --optuna-trials "$OPTUNA_TRIALS"
  --optuna-study-name "$OPTUNA_STUDY_NAME"
  --optuna-storage "$OPTUNA_STORAGE"
  --optuna-lr-low "$OPTUNA_LR_LOW"
  --optuna-lr-high "$OPTUNA_LR_HIGH"
  --optuna-weight-decay-low "$OPTUNA_WEIGHT_DECAY_LOW"
  --optuna-weight-decay-high "$OPTUNA_WEIGHT_DECAY_HIGH"
  --optuna-startup-trials "$OPTUNA_STARTUP_TRIALS"
  --optuna-pruner-startup-trials "$OPTUNA_PRUNER_STARTUP_TRIALS"
  --optuna-pruner-warmup-steps "$OPTUNA_PRUNER_WARMUP_STEPS"
)

# Bash 3.2-safe optional argument handling: append directly to the command.
if [[ "$MAX_TRAIN_BATCHES" != "0" ]]; then
  command+=(--max-train-batches "$MAX_TRAIN_BATCHES")
fi
if [[ "$MAX_VAL_BATCHES" != "0" ]]; then
  command+=(--max-val-batches "$MAX_VAL_BATCHES")
fi

run_command "${command[@]}"
