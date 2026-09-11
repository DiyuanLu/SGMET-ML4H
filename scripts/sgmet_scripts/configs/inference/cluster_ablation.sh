#!/usr/bin/env bash
RUNNER="cluster_ablation_5cv"
EXPERIMENT_NAME="sgmet_cluster_ablation"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-outputs/20260714_5cv_supervised_sgmet_frozen_experts_clusterDropout90}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CHECKPOINT_ROOT}/cluster_ablation_test}"
CLUSTER_CSV_NAME="${CLUSTER_CSV_NAME:-feature_clusters_biolord_v4_k8_leiden.csv}"
# Leave empty to read the summary-token count from each checkpoint.
NUM_SUMMARY_TOKENS="${NUM_SUMMARY_TOKENS:-}"
SAVE_SUMMARY_DIAGNOSTICS="${SAVE_SUMMARY_DIAGNOSTICS:-0}"
