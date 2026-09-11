#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_ROOT/lib/common.sh"

[[ $# -eq 1 ]] || fail "Usage: pretrain_5cv.sh <config.sh>"
CONFIG_PATH="$(resolve_config_path "$1")"
# shellcheck disable=SC1090
source "$CONFIG_PATH"

require_file "src/clinical_cluster_experts/pretrain_expert_reconstruction.py"
activate_conda_env
print_runtime

: "${EXPERIMENT_NAME:?}"
: "${CLUSTER_CSV_NAME:?}"
: "${NUM_SUMMARY_TOKENS:?}"

: "${CLUSTER_LRS:?Config must define CLUSTER_LRS as values ordered by cluster ID.}"
: "${CLUSTER_CAT_WEIGHTS:?Config must define CLUSTER_CAT_WEIGHTS as values ordered by cluster ID.}"

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
BASE_TOKEN_DIR="${BASE_TOKEN_DIR:-data/processed/tokenized_nhanes_v4}"
CV_DIR="${CV_DIR:-${BASE_TOKEN_DIR}/cv_splits}"
OUT_ROOT="${OUT_ROOT:-outputs/${DATE_TAG}_${EXPERIMENT_NAME}}"
TB_ROOT="${TB_ROOT:-runs/${DATE_TAG}_${EXPERIMENT_NAME}}"

NUM_CLUSTERS="${NUM_CLUSTERS:-8}"
FOLDS="${FOLDS:-0,1,2,3,4}"
CLUSTER_IDS="${CLUSTER_IDS:-0,1,2,3,4,5,6}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
D_MODEL="${D_MODEL:-64}"
N_LAYERS="${N_LAYERS:-2}"
N_HEADS="${N_HEADS:-4}"
DROPOUT="${DROPOUT:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
FEATURE_MASK_PROB_MAX="${FEATURE_MASK_PROB_MAX:-0.25}"
SCHEMA_DROPOUT_PROB_MAX="${SCHEMA_DROPOUT_PROB_MAX:-0.10}"
VAL_MASK_PASSES="${VAL_MASK_PASSES:-3}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-0}"
SEED="${SEED:-45}"
DEVICE="${DEVICE:-mps}"

mkdir -p "$OUT_ROOT" "$TB_ROOT"
save_run_manifest "$OUT_ROOT" "$CONFIG_PATH"

fold_ids=()
cluster_ids=()
cluster_lrs=()
cluster_cat_weights=()
IFS=',' read -r -a fold_ids <<< "$FOLDS"
IFS=',' read -r -a cluster_ids <<< "$CLUSTER_IDS"
IFS=',' read -r -a cluster_lrs <<< "$CLUSTER_LRS"
IFS=',' read -r -a cluster_cat_weights <<< "$CLUSTER_CAT_WEIGHTS"

[[ "${#cluster_lrs[@]}" -ge "$NUM_CLUSTERS" ]] \
  || fail "CLUSTER_LRS must contain at least $NUM_CLUSTERS entries."
[[ "${#cluster_cat_weights[@]}" -ge "$NUM_CLUSTERS" ]] \
  || fail "CLUSTER_CAT_WEIGHTS must contain at least $NUM_CLUSTERS entries."

for fold in "${fold_ids[@]}"; do
  token_dir="${CV_DIR}/fold${fold}"
  cluster_csv="${token_dir}/${CLUSTER_CSV_NAME}"
  require_file "${token_dir}/train_tokens.pt"
  require_file "$cluster_csv"

  for cluster_id in "${cluster_ids[@]}"; do
    lr="${cluster_lrs[$cluster_id]:-}"
    cat_weight="${cluster_cat_weights[$cluster_id]:-}"
    [[ -n "$lr" ]] || fail "No CLUSTER_LR configured for cluster $cluster_id."
    [[ -n "$cat_weight" ]] || fail "No CLUSTER_CAT_WEIGHT configured for cluster $cluster_id."

    out_dir="${OUT_ROOT}/fold${fold}/cluster${cluster_id}"
    tb_dir="${TB_ROOT}/fold${fold}/cluster${cluster_id}"
    mkdir -p "$out_dir" "$tb_dir"

    optional_args=()
    if [[ "$MAX_TRAIN_BATCHES" != "0" ]]; then
      optional_args+=(--max-train-batches "$MAX_TRAIN_BATCHES")
    fi
    if [[ "$MAX_VAL_BATCHES" != "0" ]]; then
      optional_args+=(--max-val-batches "$MAX_VAL_BATCHES")
    fi

    echo
    echo "======================================================================"
    echo "Experiment:             $EXPERIMENT_NAME"
    echo "Fold / cluster:         $fold / $cluster_id"
    echo "Summary tokens/expert:  $NUM_SUMMARY_TOKENS"
    echo "LR / cat weight:        $lr / $cat_weight"
    echo "Output:                 $out_dir"
    echo "======================================================================"

    command=(
      python -m src.clinical_cluster_experts.pretrain_expert_reconstruction
      --token-dir "$token_dir"
      --cluster-csv "$cluster_csv"
      --num-clusters "$NUM_CLUSTERS"
      --cluster-id "$cluster_id"
      --epochs "$EPOCHS"
      --batch-size "$BATCH_SIZE"
      --d-model "$D_MODEL"
      --num-summary-tokens "$NUM_SUMMARY_TOKENS"
      --n-layers "$N_LAYERS"
      --n-heads "$N_HEADS"
      --dropout "$DROPOUT"
      --feature-mask-prob-max "$FEATURE_MASK_PROB_MAX"
      --schema-dropout-prob-max "$SCHEMA_DROPOUT_PROB_MAX"
      --cat-loss focal
      --focal-gamma 1.0
      --categorical-weight "$cat_weight"
      --val-mask-passes "$VAL_MASK_PASSES"
      --lr "$lr"
      --weight-decay "$WEIGHT_DECAY"
      --lr-schedule cosine
      --warmup-frac 0.05
      --min-lr-ratio 0.05
      --seed "$SEED"
      --device "$DEVICE"
      --output-dir "$out_dir"
      --log-dir "$tb_dir"
      "${optional_args[@]}"
    )
    run_command "${command[@]}"
  done
done

echo
echo "Completed expert pretraining: $EXPERIMENT_NAME"
echo "Outputs: $OUT_ROOT"
