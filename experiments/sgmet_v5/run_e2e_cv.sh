#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${1:?usage: run_e2e_cv.sh TOKENIZED_NHANES_V5 [OUTPUT_ROOT]}"
OUTPUT_ROOT="${2:-outputs/sgmet_v5_literal149_e2e_k7_clinical_k10}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-mps}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$PYTHON" "$HERE/validate_package.py" "$DATASET_ROOT"

for k in 7 10; do
  if [[ "$k" == 7 ]]; then
    map_name="feature_clusters_biolord_v5_k7_leiden.csv"
  else
    map_name="feature_clusters_biolord_v5_k10_clinical.csv"
  fi
  for fold in 0 1 2 3 4; do
    token_dir="$DATASET_ROOT/cv_splits/fold${fold}"
    cluster_csv="$DATASET_ROOT/cluster_maps/$map_name"
    run_dir="$OUTPUT_ROOT/fold${fold}/k${k}"
    done_file="$run_dir/RUN_COMPLETE"

    if [[ -f "$done_file" ]]; then
      echo "skip fold=${fold} k=${k}: complete"
      continue
    fi
    if [[ -d "$run_dir" ]] && find "$run_dir" -mindepth 1 -print -quit | grep -q .; then
      echo "refusing to overwrite incomplete run: $run_dir" >&2
      exit 1
    fi

    mkdir -p "$run_dir" "runs/sgmet_v5_e2e/fold${fold}_k${k}"
    echo "start fold=${fold} k=${k}"
    PYTHONHASHSEED=42 "$PYTHON" -m src.clinical_cluster_experts.train_supervised_downstream \
      --token-dir "$token_dir" \
      --cluster-csv "$cluster_csv" \
      --num-clusters "$k" \
      --ignore-clusters "" \
      --train-experts \
      --allow-random-active-branches \
      --epochs 150 \
      --early-stopping-patience 15 \
      --selection-metric macro_auroc \
      --batch-size 128 \
      --d-model 64 \
      --expert-n-layers 2 \
      --expert-n-heads 4 \
      --fusion-n-layers 2 \
      --fusion-n-heads 4 \
      --dropout 0.1 \
      --lr 0.0008 \
      --expert-lr 0.0008 \
      --weight-decay 0.0005 \
      --expert-weight-decay 0.0005 \
      --lr-schedule cosine \
      --warmup-frac 0.05 \
      --min-lr-ratio 0.05 \
      --binary-loss focal \
      --focal-gamma 1.0 \
      --feature-dropout-prob-max 0.1 \
      --cluster-dropout-prob-max 0.9 \
      --seed 42 \
      --no-cluster-embedding \
      --device "$DEVICE" \
      --output-dir "$run_dir" \
      --log-dir "runs/sgmet_v5_e2e/fold${fold}_k${k}"
    touch "$done_file"
  done
done
